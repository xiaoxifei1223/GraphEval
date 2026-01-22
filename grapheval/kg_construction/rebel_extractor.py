# coding=utf-8
"""REBEL-based knowledge graph extraction.

This module uses the Babelscape REBEL-Large model for end-to-end
relation extraction without requiring separate NER and RE steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from grapheval.models.entities import Entity, RelationTriple


MODEL_NAME = "Babelscape/rebel-large"


@dataclass
class RebelExtractor:
    """Wrapper for REBEL relation extraction model."""

    model_name: str = MODEL_NAME
    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Load the REBEL model and tokenizer."""
        import torch
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading REBEL model: {self.model_name} on device: {self.device}")
        
        # 支持本地路径和离线加载
        # 如果 model_name 是本地路径，添加 local_files_only=True 确保离线
        import os
        is_local = os.path.exists(self.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=is_local  # 本地路径时强制离线
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            local_files_only=is_local  # 本地路径时强制离线
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Model cached at: ~/.cache/huggingface/hub/ (Linux/Mac) or")
        print(f"                C:\\Users\\<用户名>\\.cache\\huggingface\\hub\\ (Windows)")

    def extract_triplets(self, text: str) -> List[dict]:
        """Extract relation triplets from text using REBEL.

        Args:
            text: Input text for relation extraction.

        Returns:
            List of dictionaries with 'head', 'type' (relation), and 'tail'.
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate triplets with optimized parameters
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,           # 增加到512以生成更多三元组
            num_beams=10,             # 增加到10以探索更多可能性
            num_return_sequences=1,
            early_stopping=False,     # 关闭早停，让模型生成更完整的输出
            length_penalty=1.0,       # 鼓励生成较长的序列
            no_repeat_ngram_size=3,   # 避免重复的n-gram
        )

        # Decode output
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Parse the generated triplets
        triplets = self._parse_rebel_output(decoded)
        return triplets

    def _parse_rebel_output(self, text: str) -> List[dict]:
        """Parse REBEL model output into structured triplets.

        REBEL output format can have multiple patterns:
        - <triplet> head <subj> tail <obj> relation
        - Multiple triplets can be chained
        """
        triplets = []
        
        # Remove special tokens
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").replace("<triplet>", "|").strip()
        
        # Find all occurrences of the pattern: something <subj> something <obj> something
        # This regex captures: head <subj> tail <obj> relation (possibly followed by another <subj>)
        pattern = r"([^<|]+)\s*<subj>\s*([^<]+)\s*<obj>\s*([^<]+?)(?=\s*(?:<subj>|\||$))"
        matches = re.finditer(pattern, text)
        
        for match in matches:
            head = match.group(1).strip()
            tail = match.group(2).strip()
            relation_type = match.group(3).strip()
            
            # Clean up by removing any pipe markers
            head = head.replace("|", "").strip()
            tail = tail.replace("|", "").strip()
            relation_type = relation_type.replace("|", "").strip()
            
            if head and tail and relation_type:
                triplets.append({
                    "head": head,
                    "type": relation_type,
                    "tail": tail,
                })
        
        return triplets

    def extract_relations(self, text: str) -> List[RelationTriple]:
        """Extract relations and convert to RelationTriple objects.

        Args:
            text: Input text.

        Returns:
            List of RelationTriple objects compatible with GraphEval framework.
        """
        raw_triplets = self.extract_triplets(text)
        
        # Convert to Entity and RelationTriple objects
        triples: List[RelationTriple] = []
        entity_cache: dict[str, Entity] = {}
        
        for triplet in raw_triplets:
            head_text = triplet["head"]
            tail_text = triplet["tail"]
            relation = triplet["type"]
            
            # Reuse or create Entity objects
            if head_text not in entity_cache:
                entity_cache[head_text] = Entity(text=head_text)
            if tail_text not in entity_cache:
                entity_cache[tail_text] = Entity(text=tail_text)
            
            head_entity = entity_cache[head_text]
            tail_entity = entity_cache[tail_text]
            
            triples.append(
                RelationTriple(
                    head=head_entity,
                    relation=relation,
                    tail=tail_entity,
                    confidence=1.0,  # REBEL doesn't provide confidence scores
                )
            )
        
        return triples


def extract_kg_with_rebel(text: str, extractor: Optional[RebelExtractor] = None) -> List[RelationTriple]:
    """High-level helper to extract KG using REBEL model.

    Args:
        text: Input text for knowledge graph construction.
        extractor: Optional pre-initialized RebelExtractor.

    Returns:
        List of RelationTriple objects.

    Example:
        >>> text = "Barack Obama was born in Hawaii. He was the 44th President of the United States."
        >>> triples = extract_kg_with_rebel(text)
        >>> for triple in triples:
        ...     print(f"{triple.head.text} --[{triple.relation}]--> {triple.tail.text}")
    """
    if extractor is None:
        extractor = RebelExtractor()

    return extractor.extract_relations(text)

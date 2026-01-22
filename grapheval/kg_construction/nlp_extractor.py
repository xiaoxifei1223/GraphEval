# coding=utf-8
"""NLP-based knowledge graph extraction using DeepKE relation extraction model.

This module uses a pre-trained DeepKE model from Hugging Face to extract
relation triples from text without relying on LLM backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

from grapheval.models.entities import Entity, RelationTriple


# DeepKE 文档级关系抽取模型
MODEL_NAME = "zjunlp/deepke-re-docred-bert-base-uncased"


@dataclass
class DeepKEExtractor:
    """Wrapper for DeepKE relation extraction model."""

    model_name: str = MODEL_NAME
    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Load the model and tokenizer after initialization."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading DeepKE model: {self.model_name} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # 模型下载后的缓存位置
        cache_dir = self.tokenizer.name_or_path
        print(f"Model cached at: {torch.hub.get_dir()}/transformers/")
        print(f"Typical location: ~/.cache/huggingface/hub/ (Linux/Mac) or")
        print(f"                  C:\\Users\\<用户名>\\.cache\\huggingface\\hub\\ (Windows)")

    def extract_relations(self, text: str, entities: Optional[List[Entity]] = None) -> List[RelationTriple]:
        """Extract relation triples from text.

        Args:
            text: Input text for relation extraction.
            entities: Optional list of pre-identified entities.
                     If None, this method will attempt simple entity detection.

        Returns:
            List of RelationTriple objects.
        """
        # TODO: 实现具体的关系抽取逻辑
        # DeepKE DocRED 模型通常需要：
        # 1. 文档分句
        # 2. 实体标注（entity mentions + entity types）
        # 3. 实体对候选生成
        # 4. 对每个实体对做关系分类

        # 这里提供简化版示例框架
        triples: List[RelationTriple] = []

        # 简单示例：如果已有 entities，生成候选实体对
        if entities and len(entities) >= 2:
            # 对所有实体对做关系预测（简化版）
            for i, head_ent in enumerate(entities):
                for tail_ent in entities[i + 1:]:
                    # 构造输入（这里需要根据 DeepKE 模型的具体输入格式调整）
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        predicted_class = torch.argmax(logits, dim=-1).item()

                    # 根据预测的类别 ID 映射到关系标签（需要 label2id 映射）
                    # 这里用占位符，实际需要从模型 config 中获取
                    relation_label = f"relation_{predicted_class}"

                    if predicted_class > 0:  # 假设 0 是 "no_relation"
                        triples.append(
                            RelationTriple(
                                head=head_ent,
                                relation=relation_label,
                                tail=tail_ent,
                                confidence=0.8,  # 可以从 softmax(logits) 中计算
                            )
                        )

        return triples


# 简单的实体识别（占位符，实际应使用 spaCy 或其他 NER）
def simple_entity_detection(text: str) -> List[Entity]:
    """Placeholder for entity detection. Replace with spaCy NER or similar."""
    # 示例：简单按空格分词，找大写开头的词作为实体
    words = text.split()
    entities = []
    for word in words:
        if word and word[0].isupper():
            entities.append(Entity(text=word, type="UNKNOWN"))
    return entities


def extract_kg_with_deepke(text: str, extractor: Optional[DeepKEExtractor] = None) -> List[RelationTriple]:
    """High-level helper to extract KG using DeepKE model.

    Args:
        text: Input text.
        extractor: Optional pre-initialized DeepKEExtractor.

    Returns:
        List of relation triples.
    """
    if extractor is None:
        extractor = DeepKEExtractor()

    # Step 1: Entity detection (simplified)
    entities = simple_entity_detection(text)

    # Step 2: Relation extraction
    triples = extractor.extract_relations(text, entities)

    return triples

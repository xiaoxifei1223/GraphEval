"""Hallucination correction using an LLM backend.

This module takes hallucinated triples and a context string, and asks an
LLM to propose corrected triples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import json

from grapheval.models.entities import Entity, RelationTriple
from grapheval.kg_construction.llm_extractor import LLMClient


@dataclass
class CorrectedTriple:
    original: RelationTriple
    corrected: RelationTriple


def build_correction_prompt(triple: RelationTriple, context: str) -> str:
    """Construct a prompt asking the LLM to correct a hallucinated triple.

    The LLM is asked to return a JSON object with fields: head, relation, tail.
    """

    triple_text = f"{triple.head.text} {triple.relation} {triple.tail.text}."
    return (
        "You are a fact-checking assistant. Given a context paragraph and "
        "a possibly hallucinated fact expressed as a triple, you will propose "
        "a corrected triple that is consistent with the context.\n\n"
        "Return ONLY a JSON object with keys: head, relation, tail.\n\n"
        f"Context:\n{context}\n\n"
        f"Original triple sentence:\n{triple_text}\n"
    )


def _parse_single_corrected_triple(raw: str, original: RelationTriple) -> RelationTriple:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse corrected triple as JSON: {exc}") from exc

    head_text = str(data.get("head", original.head.text)).strip()
    relation_text = str(data.get("relation", original.relation)).strip()
    tail_text = str(data.get("tail", original.tail.text)).strip()

    head = Entity(text=head_text, type=original.head.type)
    tail = Entity(text=tail_text, type=original.tail.type)
    return RelationTriple(head=head, relation=relation_text, tail=tail)


def correct_hallucinations(
    hallucinations: Sequence[RelationTriple],
    context: str,
    llm_client: LLMClient,
) -> List[CorrectedTriple]:
    """Correct hallucinated triples using an LLM backend."""

    results: List[CorrectedTriple] = []
    for triple in hallucinations:
        prompt = build_correction_prompt(triple, context)
        raw = llm_client.complete(prompt)
        corrected_triple = _parse_single_corrected_triple(raw, triple)
        results.append(CorrectedTriple(original=triple, corrected=corrected_triple))

    return results

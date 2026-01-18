"""LLM-based information extraction for KG construction.

This module uses an LLM (e.g. Azure OpenAI) to extract entities and
relation triples from raw LLM output text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol
import json

from grapheval.models.entities import Entity, RelationTriple


class LLMClient(Protocol):
    """Abstract LLM client interface used by the extractor.

    A concrete implementation should wrap Azure OpenAI or other providers.
    """

    def complete(self, prompt: str) -> str:
        ...


@dataclass
class ExtractionResult:
    entities: List[Entity]
    triples: List[RelationTriple]


EXTRACTION_SYSTEM_PROMPT = (
    "You are an information extraction system. "
    "Given an LLM answer, you will extract named entities and "
    "semantic relations between them as JSON."
)


def build_extraction_prompt(llm_output: str) -> str:
    """Construct a prompt asking the LLM to output entities and triples.

    The expected JSON schema is:
    {
      "entities": [{"text": str, "type": str}],
      "triples": [{"head": str, "relation": str, "tail": str}]
    }
    """

    return (
        f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
        "Return a JSON object with the following keys: 'entities' and 'triples'.\n"
        "- 'entities' is a list of objects with fields: text, type.\n"
        "- 'triples' is a list of objects with fields: head, relation, tail.\n\n"
        "Answer text:\n" + llm_output
    )


def parse_extraction_response(response_text: str) -> ExtractionResult:
    """Parse the LLM JSON response into Entity and RelationTriple objects."""

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse extraction response as JSON: {exc}") from exc

    entities_by_text: dict[str, Entity] = {}

    entities_data = data.get("entities", [])
    for item in entities_data:
        text = item.get("text", "").strip()
        if not text:
            continue
        ent_type = item.get("type")
        if text not in entities_by_text:
            entities_by_text[text] = Entity(text=text, type=ent_type)

    triples: List[RelationTriple] = []
    triples_data = data.get("triples", [])
    for item in triples_data:
        head_text = item.get("head", "").strip()
        tail_text = item.get("tail", "").strip()
        relation = item.get("relation", "").strip()
        if not head_text or not tail_text or not relation:
            continue

        head = entities_by_text.setdefault(head_text, Entity(text=head_text))
        tail = entities_by_text.setdefault(tail_text, Entity(text=tail_text))
        triples.append(RelationTriple(head=head, relation=relation, tail=tail))

    return ExtractionResult(entities=list(entities_by_text.values()), triples=triples)


def extract_kg_with_llm(llm_output: str, llm_client: LLMClient) -> ExtractionResult:
    """High-level helper to construct KG using an LLM backend.

    This corresponds to the construct_kg function in the design document.
    """

    prompt = build_extraction_prompt(llm_output)
    raw_response = llm_client.complete(prompt)
    return parse_extraction_response(raw_response)

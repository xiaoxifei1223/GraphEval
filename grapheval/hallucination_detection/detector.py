"""Hallucination detection logic based on NLI outputs.

This module takes relation triples and a context string, calls an NLI client
and decides which triples are hallucinated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from grapheval.models.entities import RelationTriple
from .nli_client import NLIClient, NLIResult


@dataclass
class TripleNLIJudgement:
    triple: RelationTriple
    nli_result: NLIResult
    is_hallucination: bool


def verbalize_triple(triple: RelationTriple) -> str:
    """Convert a triple into a simple hypothesis sentence."""

    return f"{triple.head.text} {triple.relation} {triple.tail.text}."


def detect_hallucinations(
    triples: Sequence[RelationTriple],
    context: str,
    nli_client: NLIClient,
    contradiction_threshold: float = 0.5,
    neutral_threshold: float = 0.5,
) -> List[TripleNLIJudgement]:
    """Detect hallucinated triples using an NLI client.

    A triple is considered hallucinated if:
      - contradiction score >= contradiction_threshold, or
      - neutral score >= neutral_threshold (i.e. not supported by context).
    """

    hypotheses = [verbalize_triple(t) for t in triples]
    pairs = [(context, h) for h in hypotheses]
    results = nli_client.classify_batch(pairs)

    judgements: List[TripleNLIJudgement] = []
    for triple, result in zip(triples, results):
        scores: Dict[str, float] = result.scores  # type: ignore[assignment]
        contradiction_score = scores.get("contradiction", 0.0)
        neutral_score = scores.get("neutral", 0.0)
        is_hallucination = (
            contradiction_score >= contradiction_threshold
            or neutral_score >= neutral_threshold
        )
        judgements.append(
            TripleNLIJudgement(
                triple=triple,
                nli_result=result,
                is_hallucination=is_hallucination,
            )
        )

    return judgements

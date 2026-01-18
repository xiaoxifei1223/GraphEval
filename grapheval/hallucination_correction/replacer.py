"""Text-level replacement utilities for corrected triples.

This implements a simple baseline strategy: verbalize each triple as
"head relation tail" and replace it in the original output.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from grapheval.models.entities import RelationTriple


def triple_to_sentence(triple: RelationTriple) -> str:
    return f"{triple.head.text} {triple.relation} {triple.tail.text}"


def replace_triples(
    original_output: str,
    hallucinations: Sequence[RelationTriple],
    corrected_triples: Sequence[RelationTriple],
) -> str:
    """Replace hallucinated triples in the original output with corrected ones.

    This is a naive string-based implementation that may be refined later.
    """

    if len(hallucinations) != len(corrected_triples):
        raise ValueError("hallucinations and corrected_triples must have the same length")

    corrected_output = original_output
    for old_triple, new_triple in zip(hallucinations, corrected_triples):
        old_sentence = triple_to_sentence(old_triple)
        new_sentence = triple_to_sentence(new_triple)
        if not old_sentence:
            continue
        corrected_output = corrected_output.replace(old_sentence, new_sentence)

    return corrected_output

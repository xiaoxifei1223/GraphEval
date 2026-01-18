"""Core data models for GraphEval entities and relation triples."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MentionSpan:
    """Represents a span of text in the original LLM output.

    start and end are character offsets (Python slicing semantics).
    """

    start: int
    end: int


@dataclass
class Entity:
    """An entity detected in the LLM output."""

    text: str
    type: Optional[str] = None
    id: Optional[str] = None
    mentions: List[MentionSpan] = field(default_factory=list)


@dataclass
class RelationTriple:
    """A semantic relation between two entities.

    This mirrors the (head, relation, tail) triple used in the design.
    """

    head: Entity
    relation: str
    tail: Entity
    confidence: float = 1.0

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.head.text, self.relation, self.tail.text

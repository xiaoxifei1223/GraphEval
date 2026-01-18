"""GraphEval pipeline orchestrating KG construction, detection and correction.

The main entry point is the GraphEvalPipeline class with a run() method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from grapheval.kg_construction.llm_extractor import (
    LLMClient,
    ExtractionResult,
    extract_kg_with_llm,
)
from grapheval.hallucination_detection.nli_client import NLIClient
from grapheval.hallucination_detection.detector import (
    TripleNLIJudgement,
    detect_hallucinations,
)
from grapheval.hallucination_correction.corrector import (
    CorrectedTriple,
    correct_hallucinations,
)
from grapheval.hallucination_correction.replacer import replace_triples


@dataclass
class PipelineResult:
    original_output: str
    kg_triples: List[Dict[str, Any]]
    hallucinated_triples: List[Dict[str, Any]]
    corrected_triples: List[Dict[str, Any]]
    corrected_output: str


class GraphEvalPipeline:
    """High-level orchestrator for the GraphEval process."""

    def __init__(
        self,
        llm_client: LLMClient,
        nli_client: Optional[NLIClient] = None,
    ) -> None:
        self._llm_client = llm_client
        self._nli_client = nli_client or NLIClient()

    @staticmethod
    def _triple_to_dict(triple) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        return {
            "head": triple.head.text,
            "relation": triple.relation,
            "tail": triple.tail.text,
        }

    @staticmethod
    def _judgement_to_dict(j: TripleNLIJudgement) -> Dict[str, Any]:
        return {
            "triple": GraphEvalPipeline._triple_to_dict(j.triple),
            "label": j.nli_result.label,
            "scores": j.nli_result.scores,
            "is_hallucination": j.is_hallucination,
        }

    @staticmethod
    def _corrected_to_dict(c: CorrectedTriple) -> Dict[str, Any]:
        return {
            "original": GraphEvalPipeline._triple_to_dict(c.original),
            "corrected": GraphEvalPipeline._triple_to_dict(c.corrected),
        }

    def run(self, llm_output: str, context: str) -> PipelineResult:
        """Run the full GraphEval pipeline.

        Steps:
          1. Construct KG from llm_output.
          2. Detect hallucinated triples with NLI.
          3. Correct hallucinated triples with LLM and replace them in the output.
        """

        # 1. KG construction
        extraction: ExtractionResult = extract_kg_with_llm(llm_output, self._llm_client)
        triples = extraction.triples

        # 2. Hallucination detection
        judgements: List[TripleNLIJudgement] = detect_hallucinations(
            triples=triples,
            context=context,
            nli_client=self._nli_client,
        )
        hallucinated_triples = [j.triple for j in judgements if j.is_hallucination]

        # 3. Hallucination correction
        corrected_triple_objs: List[CorrectedTriple] = []
        corrected_output = llm_output
        if hallucinated_triples:
            corrected_triple_objs = correct_hallucinations(
                hallucinations=hallucinated_triples,
                context=context,
                llm_client=self._llm_client,
            )
            corrected_only = [c.corrected for c in corrected_triple_objs]
            corrected_output = replace_triples(
                original_output=llm_output,
                hallucinations=hallucinated_triples,
                corrected_triples=corrected_only,
            )

        kg_triples_dicts = [self._triple_to_dict(t) for t in triples]
        hallucinated_dicts = [self._judgement_to_dict(j) for j in judgements if j.is_hallucination]
        corrected_dicts = [self._corrected_to_dict(c) for c in corrected_triple_objs]

        return PipelineResult(
            original_output=llm_output,
            kg_triples=kg_triples_dicts,
            hallucinated_triples=hallucinated_dicts,
            corrected_triples=corrected_dicts,
            corrected_output=corrected_output,
        )

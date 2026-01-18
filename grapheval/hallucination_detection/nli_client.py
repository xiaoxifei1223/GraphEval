"""NLI model client for hallucination detection.

This module wraps a Hugging Face transformers NLI model
and exposes a small interface for classifying premise/hypothesis pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple
import json

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from grapheval.kg_construction.llm_extractor import LLMClient


NLI_LABEL = Literal["entailment", "contradiction", "neutral"]


@dataclass
class NLIResult:
    label: NLI_LABEL
    scores: Dict[NLI_LABEL, float]


class NLIClient:
    """Thin wrapper around a transformers NLI model."""

    def __init__(self, model_name_or_path: str = "facebook/bart-large-mnli", device: str | None = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self._pipe = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            return_all_scores=True,
            device=device if device is not None else -1,
        )

        # Map model-specific labels (e.g. CONTRADICTION) to canonical ones.
        self._label_mapping = {
            "entailment": "entailment",
            "ENTAILMENT": "entailment",
            "contradiction": "contradiction",
            "CONTRADICTION": "contradiction",
            "neutral": "neutral",
            "NEUTRAL": "neutral",
        }

    def _canonical_label(self, label: str) -> NLI_LABEL:
        mapped = self._label_mapping.get(label, label.lower())
        if mapped not in ("entailment", "contradiction", "neutral"):
            raise ValueError(f"Unexpected NLI label: {label}")
        return mapped  # type: ignore[return-value]

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify a single premise/hypothesis pair."""

        return self.classify_batch([(premise, hypothesis)])[0]

    def classify_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIResult]:
        """Batch classification for multiple premise/hypothesis pairs."""

        inputs = [{"text": p, "text_pair": h} for p, h in pairs]
        outputs: List[List[Dict[str, float]]] = self._pipe(inputs)  # type: ignore[assignment]

        results: List[NLIResult] = []
        for scores_for_pair in outputs:
            score_map: Dict[NLI_LABEL, float] = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            for item in scores_for_pair:
                raw_label = item["label"]
                prob = float(item["score"])
                canonical = self._canonical_label(raw_label)
                score_map[canonical] = prob

            # Pick best label by probability
            best_label = max(score_map.items(), key=lambda kv: kv[1])[0]
            results.append(NLIResult(label=best_label, scores=score_map))

        return results


LLM_NLI_SYSTEM_PROMPT = (
    "You are a natural language inference (NLI) classifier. "
    "Given a premise and a hypothesis, you must decide whether the "
    "hypothesis is ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to "
    "the premise. Respond ONLY with a JSON object of the form "
    '{"label": "entailment" | "contradiction" | "neutral"}.\n'
)


class LLMNLIClient:
    """Use a generic LLM backend to perform NLI classification.

    This client relies on an LLMClient implementation (e.g. Azure OpenAI)
    to classify premise/hypothesis pairs. It outputs the same NLIResult
    structure as the transformer-based NLIClient.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._label_mapping = {
            "entailment": "entailment",
            "ENTAILMENT": "entailment",
            "contradiction": "contradiction",
            "CONTRADICTION": "contradiction",
            "neutral": "neutral",
            "NEUTRAL": "neutral",
        }

    def _canonical_label(self, label: str) -> NLI_LABEL:
        mapped = self._label_mapping.get(label, label.lower())
        if mapped not in ("entailment", "contradiction", "neutral"):
            raise ValueError(f"Unexpected NLI label from LLM: {label}")
        return mapped  # type: ignore[return-value]

    def _build_prompt(self, premise: str, hypothesis: str) -> str:
        return (
            f"{LLM_NLI_SYSTEM_PROMPT}\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
        )

    def _parse_response(self, response_text: str) -> NLIResult:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM NLI response as JSON: {exc}") from exc

        raw_label = str(data.get("label", "")).strip()
        if not raw_label:
            raise ValueError("LLM NLI response missing 'label' field")

        canonical = self._canonical_label(raw_label)
        scores: Dict[NLI_LABEL, float] = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        scores[canonical] = 1.0
        return NLIResult(label=canonical, scores=scores)

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        prompt = self._build_prompt(premise, hypothesis)
        raw = self._llm_client.complete(prompt)
        return self._parse_response(raw)

    def classify_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIResult]:
        # For simplicity, call the LLM sequentially for each pair.
        return [self.classify(p, h) for p, h in pairs]

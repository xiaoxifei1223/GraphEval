"""Simple CLI entrypoint for running the GraphEval pipeline.

Usage (example):
    python -m cli.main --llm-output "..." --context "..."

Note: You must provide an implementation of LLMClient that talks to
Azure OpenAI or another provider and pass it into GraphEvalPipeline.
This CLI uses a placeholder stub and will raise at runtime until you
replace it with a real client.
"""
from __future__ import annotations

import argparse
from typing import Any

from grapheval.pipeline.pipeline import GraphEvalPipeline
from grapheval.kg_construction.llm_extractor import LLMClient


class StubLLMClient:
    """Placeholder LLM client.

    Replace this with a real implementation that calls Azure OpenAI.
    """

    def complete(self, prompt: str) -> str:  # type: ignore[override]
        raise NotImplementedError("StubLLMClient.complete must be replaced with a real implementation")


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Run GraphEval pipeline on LLM output and context.")
    parser.add_argument("--llm-output", required=True, help="LLM answer text to be evaluated")
    parser.add_argument("--context", required=True, help="Reference context text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm_client: LLMClient = StubLLMClient()
    pipeline = GraphEvalPipeline(llm_client=llm_client)

    result = pipeline.run(llm_output=args.llm_output, context=args.context)
    # For now we print the result dictionary representation.
    print(result)


if __name__ == "__main__":
    main()

"""Global configuration and settings for GraphEval.

This module defines configuration objects for external services
such as Azure OpenAI and Hugging Face NLI models.
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class AzureOpenAISettings:
    endpoint: str
    api_key: str
    deployment_name: str


@dataclass
class NLIModelSettings:
    model_name_or_path: str = "facebook/bart-large-mnli"
    device: Optional[str] = None  # e.g. "cpu" or "cuda"


@dataclass
class GraphEvalSettings:
    azure_openai: AzureOpenAISettings
    nli_model: NLIModelSettings = NLIModelSettings()


def load_settings() -> GraphEvalSettings:
    """Load settings from environment variables.

    This keeps secrets (keys, endpoints) outside of the codebase.
    """

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

    if not endpoint or not api_key or not deployment_name:
        raise RuntimeError(
            "Azure OpenAI configuration is incomplete. "
            "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "and AZURE_OPENAI_DEPLOYMENT environment variables."
        )

    azure_settings = AzureOpenAISettings(
        endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
    )

    nli_model_name = os.getenv("GRAPHEVAL_NLI_MODEL", "facebook/bart-large-mnli")
    nli_device = os.getenv("GRAPHEVAL_NLI_DEVICE") or None

    nli_settings = NLIModelSettings(
        model_name_or_path=nli_model_name,
        device=nli_device,
    )

    return GraphEvalSettings(azure_openai=azure_settings, nli_model=nli_settings)

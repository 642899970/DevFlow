"""MiMo Orbit adapter for LLM interface."""

from typing import AsyncIterator, Dict, List, Optional

from devflow.core.models.openai import OpenAIModel


class MiMoModel(OpenAIModel):
    """MiMo Orbit model adapter (OpenAI-compatible)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """Initialize the MiMo model.

        MiMo Orbit provides an OpenAI-compatible API, so we extend OpenAIModel.

        Args:
            model_name: Name of the MiMo model (e.g., mimo-v2.5-pro)
            api_key: MiMo API key
            base_url: MiMo API base URL (default: https://api.mimo.xiaomi.com/v1)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        # Set default base URL for MiMo if not provided
        if base_url is None:
            base_url = "https://api.mimo.xiaomi.com/v1"

        super().__init__(model_name, api_key, base_url, temperature, max_tokens, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the MiMo model.

        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info["provider"] = "mimo"
        return info

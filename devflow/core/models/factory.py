"""Factory for creating model instances."""

import os
from typing import Dict, Optional

from devflow.core.models.anthropic import AnthropicModel
from devflow.core.models.base import BaseLLM
from devflow.core.models.mimo import MiMoModel
from devflow.core.models.openai import OpenAIModel
from devflow.exceptions.errors import (
    ModelInitializationError,
    ModelNotFoundError,
)


class ModelFactory:
    """Factory for creating model instances."""

    _models: Dict[str, BaseLLM] = {}
    _model_configs: Dict[str, Dict] = {}

    @classmethod
    def register_model_config(cls, model_id: str, config: Dict) -> None:
        """Register a model configuration.

        Args:
            model_id: Unique identifier for the model
            config: Model configuration dictionary
        """
        cls._model_configs[model_id] = config

    @classmethod
    def register_model(cls, model_id: str, model: BaseLLM) -> None:
        """Register a model instance.

        Args:
            model_id: Unique identifier for the model
            model: Model instance
        """
        cls._models[model_id] = model

    @classmethod
    def get_model(cls, model_id: str) -> Optional[BaseLLM]:
        """Get a registered model by ID.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Model instance or None if not found
        """
        return cls._models.get(model_id)

    @classmethod
    def create_model(cls, model_id: str, config: Optional[Dict] = None) -> BaseLLM:
        """Create a model instance from configuration.

        Args:
            model_id: Unique identifier for the model
            config: Model configuration (uses registered config if not provided)

        Returns:
            Model instance

        Raises:
            ModelNotFoundError: If model configuration is not found
            ModelInitializationError: If model creation fails
        """
        if config is None:
            config = cls._model_configs.get(model_id)
            if config is None:
                raise ModelNotFoundError(f"Model configuration not found for: {model_id}")

        provider = config.get("provider")
        model_name = config.get("model_name")
        api_key_env = config.get("api_key_env")
        base_url = config.get("base_url")
        default_params = config.get("default_params", {})

        # Get API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ModelInitializationError(
                f"API key not found in environment variable: {api_key_env}"
            )

        # Create model based on provider
        try:
            if provider == "openai":
                model = OpenAIModel(
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=default_params.get("temperature", 0.7),
                    max_tokens=default_params.get("max_tokens", 4096),
                )
            elif provider == "anthropic":
                model = AnthropicModel(
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=default_params.get("temperature", 0.7),
                    max_tokens=default_params.get("max_tokens", 4096),
                )
            elif provider == "mimo" or provider == "openai_compatible":
                model = MiMoModel(
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=default_params.get("temperature", 0.7),
                    max_tokens=default_params.get("max_tokens", 4096),
                )
            else:
                raise ModelInitializationError(f"Unknown provider: {provider}")

            # Register the model
            cls.register_model(model_id, model)
            return model

        except Exception as e:
            raise ModelInitializationError(f"Failed to create model {model_id}: {str(e)}")

    @classmethod
    def create_all_models(cls, configs: Dict[str, Dict]) -> Dict[str, BaseLLM]:
        """Create all models from configurations.

        Args:
            configs: Dictionary of model configurations keyed by model_id

        Returns:
            Dictionary of created model instances
        """
        models = {}
        for model_id, config in configs.items():
            try:
                model = cls.create_model(model_id, config)
                models[model_id] = model
            except ModelInitializationError as e:
                print(f"Warning: Failed to create model {model_id}: {str(e)}")
        return models

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models and configurations."""
        cls._models.clear()
        cls._model_configs.clear()

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model IDs.

        Returns:
            List of model IDs
        """
        return list(cls._models.keys())

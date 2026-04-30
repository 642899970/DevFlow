"""Configuration management for DevFlow."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from devflow.exceptions.errors import ConfigurationError
from devflow.schemas.config import (
    AgentConfig,
    AgentsConfig,
    DevFlowConfig,
    ModelConfig,
    ModelsConfig,
)


class ConfigManager:
    """Manager for loading and validating configuration files."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to config/ directory relative to current working directory
            config_dir = Path.cwd() / "config"

        self.config_dir = Path(config_dir)
        self._config: Optional[DevFlowConfig] = None
        self._models_config: Optional[ModelsConfig] = None
        self._agents_config: Optional[AgentsConfig] = None

    def load_config(self) -> DevFlowConfig:
        """Load the main configuration.

        Returns:
            DevFlowConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._config is not None:
            return self._config

        try:
            models_config = self.load_models_config()
            agents_config = self.load_agents_config()

            self._config = DevFlowConfig(
                models=models_config,
                agents=agents_config,
            )
            return self._config

        except (ValidationError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    def load_models_config(self) -> ModelsConfig:
        """Load models configuration from models.yaml.

        Returns:
            ModelsConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._models_config is not None:
            return self._models_config

        models_file = self.config_dir / "models.yaml"

        if not models_file.exists():
            raise ConfigurationError(f"Models configuration file not found: {models_file}")

        try:
            with open(models_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self._models_config = ModelsConfig(**data)
            return self._models_config

        except (ValidationError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Failed to load models configuration: {str(e)}")

    def load_agents_config(self) -> AgentsConfig:
        """Load agents configuration from agents.yaml.

        Returns:
            AgentsConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._agents_config is not None:
            return self._agents_config

        agents_file = self.config_dir / "agents.yaml"

        if not agents_file.exists():
            raise ConfigurationError(f"Agents configuration file not found: {agents_file}")

        try:
            with open(agents_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Convert agents list to dict
            agents_dict = {}
            if "agents" in data:
                for agent_name, agent_data in data["agents"].items():
                    agents_dict[agent_name] = AgentConfig(
                        name=agent_data.get("name", agent_name),
                        model=agent_data["model"],
                        system_prompt=agent_data["system_prompt"],
                        tools=agent_data.get("tools", []),
                        max_iterations=agent_data.get("max_iterations", 10),
                    )

            self._agents_config = AgentsConfig(agents=agents_dict)
            return self._agents_config

        except (ValidationError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Failed to load agents configuration: {str(e)}")

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig or None if not found
        """
        models_config = self.load_models_config()
        return models_config.get_model(model_id)

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get an agent configuration by name.

        Args:
            agent_name: Agent name

        Returns:
            AgentConfig or None if not found
        """
        agents_config = self.load_agents_config()
        return agents_config.get_agent(agent_name)

    def get_all_model_ids(self) -> list[str]:
        """Get all model IDs.

        Returns:
            List of model IDs
        """
        models_config = self.load_models_config()
        return models_config.get_all_ids()

    def get_all_agent_names(self) -> list[str]:
        """Get all agent names.

        Returns:
            List of agent names
        """
        agents_config = self.load_agents_config()
        return agents_config.get_all_names()

    def validate_environment(self) -> Dict[str, bool]:
        """Validate that all required environment variables are set.

        Returns:
            Dictionary mapping environment variable names to availability status
        """
        models_config = self.load_models_config()
        results = {}

        for model in models_config.models:
            env_var = model.api_key_env
            results[env_var] = os.getenv(env_var) is not None

        return results

    def reload(self) -> None:
        """Reload all configurations from files."""
        self._config = None
        self._models_config = None
        self._agents_config = None

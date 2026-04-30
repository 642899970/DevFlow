"""Pydantic models for configuration."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ModelParams(BaseModel):
    """Default parameters for a model."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    id: str = Field(..., description="Unique identifier for the model")
    provider: str = Field(..., description="Provider: openai, anthropic, mimo, openai_compatible")
    model_name: str = Field(..., description="Model name as used by the provider")
    api_key_env: str = Field(..., description="Environment variable name for API key")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API")
    default_params: ModelParams = Field(default_factory=ModelParams)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = {"openai", "anthropic", "mimo", "openai_compatible"}
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v


class ModelsConfig(BaseModel):
    """Configuration for all models."""

    models: List[ModelConfig] = Field(default_factory=list)

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID."""
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def get_all_ids(self) -> List[str]:
        """Get all model IDs."""
        return [model.id for model in self.models]


class ToolConfig(BaseModel):
    """Configuration for a tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str = Field(..., description="Agent name")
    model: str = Field(..., description="Model ID to use")
    system_prompt: str = Field(..., description="System prompt for the agent")
    tools: List[str] = Field(default_factory=list, description="List of tool names")
    max_iterations: int = Field(default=10, gt=0, description="Maximum iterations")


class AgentsConfig(BaseModel):
    """Configuration for all agents."""

    agents: Dict[str, AgentConfig] = Field(default_factory=dict)

    def get_agent(self, agent_name: str) -> Optional[AgentConfig]:
        """Get an agent configuration by name."""
        return self.agents.get(agent_name)

    def get_all_names(self) -> List[str]:
        """Get all agent names."""
        return list(self.agents.keys())


class DevFlowConfig(BaseModel):
    """Main configuration for DevFlow."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DevFlowConfig":
        """Create configuration from dictionary."""
        return cls(**data)

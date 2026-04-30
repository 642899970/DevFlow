"""Token usage tracker for DevFlow."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage record."""

    model_id: str = Field(..., description="Model identifier")
    agent_name: str = Field(..., description="Agent name")
    prompt_tokens: int = Field(default=0, description="Prompt tokens used")
    completion_tokens: int = Field(default=0, description="Completion tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp")


class TokenTracker:
    """Tracker for token usage across models and agents."""

    def __init__(self):
        """Initialize the token tracker."""
        self._usage: List[TokenUsage] = []
        self._by_model: Dict[str, List[TokenUsage]] = defaultdict(list)
        self._by_agent: Dict[str, List[TokenUsage]] = defaultdict(list)

    def record_usage(
        self,
        model_id: str,
        agent_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Record token usage.

        Args:
            model_id: Model identifier
            agent_name: Agent name
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            total_tokens: Total tokens used (calculated if not provided)
        """
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        usage = TokenUsage(
            model_id=model_id,
            agent_name=agent_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        self._usage.append(usage)
        self._by_model[model_id].append(usage)
        self._by_agent[agent_name].append(usage)

    def get_total_tokens(self) -> int:
        """Get total tokens used across all models and agents.

        Returns:
            Total tokens used
        """
        return sum(u.total_tokens for u in self._usage)

    def get_tokens_by_model(self, model_id: str) -> int:
        """Get total tokens used by a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Total tokens used by the model
        """
        return sum(u.total_tokens for u in self._by_model.get(model_id, []))

    def get_tokens_by_agent(self, agent_name: str) -> int:
        """Get total tokens used by a specific agent.

        Args:
            agent_name: Agent name

        Returns:
            Total tokens used by the agent
        """
        return sum(u.total_tokens for u in self._by_agent.get(agent_name, []))

    def get_model_summary(self) -> Dict[str, Dict[str, int]]:
        """Get a summary of token usage by model.

        Returns:
            Dictionary mapping model IDs to token usage statistics
        """
        summary = {}
        for model_id, usages in self._by_model.items():
            summary[model_id] = {
                "prompt_tokens": sum(u.prompt_tokens for u in usages),
                "completion_tokens": sum(u.completion_tokens for u in usages),
                "total_tokens": sum(u.total_tokens for u in usages),
                "call_count": len(usages),
            }
        return summary

    def get_agent_summary(self) -> Dict[str, Dict[str, int]]:
        """Get a summary of token usage by agent.

        Returns:
            Dictionary mapping agent names to token usage statistics
        """
        summary = {}
        for agent_name, usages in self._by_agent.items():
            summary[agent_name] = {
                "prompt_tokens": sum(u.prompt_tokens for u in usages),
                "completion_tokens": sum(u.completion_tokens for u in usages),
                "total_tokens": sum(u.total_tokens for u in usages),
                "call_count": len(usages),
            }
        return summary

    def get_all_usage(self) -> List[TokenUsage]:
        """Get all token usage records.

        Returns:
            List of all token usage records
        """
        return self._usage.copy()

    def clear(self) -> None:
        """Clear all token usage records."""
        self._usage.clear()
        self._by_model.clear()
        self._by_agent.clear()

    def save_to_file(self, file_path: Path) -> None:
        """Save token usage to a JSON file.

        Args:
            file_path: Path to save the file to
        """
        import json

        data = {
            "total_tokens": self.get_total_tokens(),
            "by_model": self.get_model_summary(),
            "by_agent": self.get_agent_summary(),
            "usage_records": [u.model_dump() for u in self._usage],
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "TokenTracker":
        """Load token usage from a JSON file.

        Args:
            file_path: Path to load the file from

        Returns:
            TokenTracker instance with loaded data
        """
        import json

        tracker = cls()

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for record in data.get("usage_records", []):
            tracker.record_usage(
                model_id=record["model_id"],
                agent_name=record["agent_name"],
                prompt_tokens=record["prompt_tokens"],
                completion_tokens=record["completion_tokens"],
                total_tokens=record["total_tokens"],
            )

        return tracker

    def estimate_cost(self, model_costs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Estimate the cost of token usage.

        Args:
            model_costs: Dictionary mapping model IDs to cost per 1K tokens
                        {"prompt": 0.01, "completion": 0.03}

        Returns:
            Dictionary mapping model IDs to estimated costs
        """
        costs = {}
        for model_id, usages in self._by_model.items():
            if model_id not in model_costs:
                continue

            costs_per_1k = model_costs[model_id]
            prompt_cost = (sum(u.prompt_tokens for u in usages) / 1000) * costs_per_1k.get("prompt", 0)
            completion_cost = (
                (sum(u.completion_tokens for u in usages) / 1000) * costs_per_1k.get("completion", 0)
            )

            costs[model_id] = prompt_cost + completion_cost

        return costs

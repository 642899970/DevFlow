# DevFlow

Multi-agent collaborative full-stack development platform.

## Overview

DevFlow is an AI-powered development platform that uses multiple specialized agents to automatically complete architecture design, frontend/backend code generation, and test case generation from natural language requirements.

## Features

- **Multi-LLM Support**: OpenAI, Anthropic Claude, Xiaomi MiMo, and OpenAI-compatible APIs
- **Specialized Agents**: Architect, Frontend, Backend, and Test agents
- **Task Planning**: Automatic DAG generation from natural language
- **Parallel Execution**: Concurrent task execution based on dependencies
- **Tool System**: File operations, command execution, web search
- **CLI Interface**: Interactive mode, dry-run, and resume capability
- **Token Tracking**: Per-model and per-agent usage monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/devflow.git
cd devflow

# Install dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Configuration

Set up your environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Configure models and agents in `config/models.yaml` and `config/agents.yaml`.

## Usage

```bash
# Create a new project
devflow new "Create a blog system with user authentication"

# Use custom configuration
devflow new "Create a todo app" --model-config config/models.yaml --agents config/agents.yaml

# Dry-run mode (planning only)
devflow new "Create a REST API" --dry-run

# Resume a session
devflow resume <session-id>

# Check session status
devflow status <session-id>
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=devflow

# Format code
black devflow tests

# Lint code
ruff check devflow tests

# Type check
mypy devflow
```

## Architecture

```
devflow/
├── cli/                    # Command-line interface
├── core/
│   ├── models/            # Multi-LLM Adapter Layer
│   ├── agents/            # Agent Framework
│   ├── planning/          # Task Planning Engine
│   ├── orchestration/     # Multi-Agent Orchestration
│   ├── tools/             # Tool System
│   ├── workspace/         # Shared Workspace
│   └── logging/           # Logging and Monitoring
├── schemas/               # Pydantic models
├── utils/                 # Utilities
└── exceptions/            # Custom exceptions
```

## License

MIT

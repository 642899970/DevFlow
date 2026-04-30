"""CLI entry point for DevFlow."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from devflow.core.config import ConfigManager
from devflow.core.logging.logger import StructuredLogger
from devflow.core.logging.tracker import TokenTracker
from devflow.core.models.factory import ModelFactory
from devflow.core.workspace.manager import WorkspaceManager
from devflow.exceptions.errors import ConfigurationError, DevFlowError

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """DevFlow - Multi-agent collaborative full-stack development platform."""
    pass


@cli.command()
@click.argument("requirement")
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    default="config/models.yaml",
    help="Path to models configuration file",
)
@click.option(
    "--agents-config",
    type=click.Path(exists=True),
    default="config/agents.yaml",
    help="Path to agents configuration file",
)
@click.option(
    "--output",
    type=click.Path(),
    default="output",
    help="Output directory for generated project",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate plan only without executing",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Interactive mode with step-by-step confirmation",
)
def new(requirement: str, model_config: str, agents_config: str, output: str, dry_run: bool, interactive: bool):
    """Create a new project from a natural language requirement.

    REQUIREMENT: Natural language description of the project to create
    """
    try:
        console.print(Panel(f"[bold blue]DevFlow[/bold blue]\n\nRequirement: {requirement}", title="New Project"))

        # Load configuration
        config_dir = Path(model_config).parent
        config_manager = ConfigManager(config_dir)

        console.print(f"\n[green]✓[/green] Loaded configuration from {config_dir}")

        # Validate environment
        env_status = config_manager.validate_environment()
        missing_keys = [k for k, v in env_status.items() if not v]

        if missing_keys:
            console.print(f"\n[yellow]⚠[/yellow] Missing environment variables: {', '.join(missing_keys)}")
            console.print("[yellow]Please set these variables in your .env file[/yellow]")
            sys.exit(1)

        console.print(f"[green]✓[/green] All environment variables set")

        # Initialize models
        models_config = config_manager.load_models_config()
        ModelFactory.clear()

        for model_config_data in models_config.models:
            ModelFactory.register_model_config(model_config_data.id, model_config_data.model_dump())

        console.print(f"[green]✓[/green] Registered {len(models_config.models)} models")

        # Initialize workspace
        workspace = WorkspaceManager.get_instance()
        console.print(f"[green]✓[/green] Initialized workspace at {workspace.workspace_path}")

        # Initialize logging
        logger = StructuredLogger()
        token_tracker = TokenTracker()

        # Generate task plan
        console.print("\n[bold]Generating task plan...[/bold]")

        # For now, we'll just show a placeholder
        # In a full implementation, this would use the TaskPlanner
        console.print("[yellow]⚠[/yellow] Task planning not yet implemented")
        console.print("[yellow]This is a placeholder for the planning phase[/yellow]")

        if dry_run:
            console.print("\n[bold]Dry-run mode: Plan generated, not executing[/bold]")
            return

        # Execute tasks
        console.print("\n[bold]Executing tasks...[/bold]")
        console.print("[yellow]⚠[/yellow] Task execution not yet implemented")
        console.print("[yellow]This is a placeholder for the execution phase[/yellow]")

        # Export results
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[green]✓[/green] Project exported to {output_path}")

        # Show token usage
        console.print("\n[bold]Token Usage:[/bold]")
        table = Table()
        table.add_column("Model", style="cyan")
        table.add_column("Tokens", style="green")
        table.add_column("Cost", style="yellow")

        # Placeholder data
        table.add_row("Total", "0", "$0.00")

        console.print(table)

    except ConfigurationError as e:
        console.print(f"\n[red]✗[/red] Configuration error: {str(e)}")
        sys.exit(1)
    except DevFlowError as e:
        console.print(f"\n[red]✗[/red] Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗[/red] Unexpected error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("session_id")
def status(session_id: str):
    """Show the status of a session.

    SESSION_ID: Session identifier
    """
    console.print(f"\n[bold]Session Status: {session_id}[/bold]")
    console.print("[yellow]⚠[/yellow] Session status not yet implemented")


@cli.command()
@click.argument("session_id")
def resume(session_id: str):
    """Resume a session.

    SESSION_ID: Session identifier
    """
    console.print(f"\n[bold]Resuming session: {session_id}[/bold]")
    console.print("[yellow]⚠[/yellow] Session resume not yet implemented")


@cli.command()
def config():
    """Show current configuration."""
    try:
        config_manager = ConfigManager()

        console.print("\n[bold]Models Configuration:[/bold]")
        models_config = config_manager.load_models_config()

        table = Table()
        table.add_column("ID", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("API Key Env", style="magenta")

        for model in models_config.models:
            table.add_row(model.id, model.provider, model.model_name, model.api_key_env)

        console.print(table)

        console.print("\n[bold]Agents Configuration:[/bold]")
        agents_config = config_manager.load_agents_config()

        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Tools", style="yellow")
        table.add_column("Max Iterations", style="magenta")

        for agent_name, agent_config in agents_config.agents.items():
            tools_str = ", ".join(agent_config.tools) if agent_config.tools else "None"
            table.add_row(
                agent_name,
                agent_config.model,
                tools_str,
                str(agent_config.max_iterations),
            )

        console.print(table)

    except ConfigurationError as e:
        console.print(f"\n[red]✗[/red] Configuration error: {str(e)}")
        sys.exit(1)


@cli.command()
def init():
    """Initialize DevFlow configuration files."""
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create models.yaml
    models_file = config_dir / "models.yaml"
    if not models_file.exists():
        models_file.write_text(
            """models:
  - id: gpt4o
    provider: openai
    model_name: gpt-4o
    api_key_env: OPENAI_API_KEY
    default_params:
      temperature: 0.7
      max_tokens: 4096

  - id: sonnet
    provider: anthropic
    model_name: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY
    default_params:
      temperature: 0.7
      max_tokens: 4096
"""
        )
        console.print(f"[green]✓[/green] Created {models_file}")

    # Create agents.yaml
    agents_file = config_dir / "agents.yaml"
    if not agents_file.exists():
        agents_file.write_text(
            """agents:
  architect:
    name: Architect
    model: gpt4o
    system_prompt: |
      You are a system architect responsible for designing software systems.
      Your tasks include:
      - Analyzing requirements and creating system architecture
      - Designing database schemas
      - Defining API interfaces
      - Creating project structure
      - Providing technical guidance to other agents

      Always think step by step and explain your reasoning.
      Use the available tools to document your designs.
    tools:
      - write_file
      - read_file
    max_iterations: 10

  frontend:
    name: Frontend Engineer
    model: gpt4o
    system_prompt: |
      You are a frontend engineer specializing in modern web development.
      Your tasks include:
      - Creating responsive user interfaces
      - Implementing frontend logic
      - Integrating with backend APIs
      - Writing clean, maintainable code

      Use React + TailwindCSS for UI development.
      Follow best practices for accessibility and performance.
    tools:
      - write_file
      - read_file
      - execute_command
    max_iterations: 15

  backend:
    name: Backend Engineer
    model: sonnet
    system_prompt: |
      You are a backend engineer specializing in API development.
      Your tasks include:
      - Implementing RESTful APIs
      - Database operations
      - Authentication and authorization
      - Business logic implementation

      Use FastAPI for API development.
      Follow REST principles and write clean, testable code.
    tools:
      - write_file
      - read_file
      - execute_command
    max_iterations: 15

  tester:
    name: Test Engineer
    model: gpt4o
    system_prompt: |
      You are a test engineer responsible for ensuring code quality.
      Your tasks include:
      - Writing unit tests
      - Writing integration tests
      - Creating test cases
      - Identifying edge cases and bugs

      Use pytest for testing.
      Aim for high test coverage and meaningful test cases.
    tools:
      - write_file
      - read_file
      - run_tests
    max_iterations: 10
"""
        )
        console.print(f"[green]✓[/green] Created {agents_file}")

    # Create .env.example
    env_file = Path(".env.example")
    if not env_file.exists():
        env_file.write_text(
            """# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Custom model endpoints
# OPENAI_BASE_URL=https://api.openai.com/v1
# ANTHROPIC_BASE_URL=https://api.anthropic.com
"""
        )
        console.print(f"[green]✓[/green] Created {env_file}")

    console.print("\n[green]✓[/green] Configuration initialized successfully!")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Copy .env.example to .env and add your API keys")
    console.print("2. Edit config/models.yaml and config/agents.yaml as needed")
    console.print("3. Run: devflow new \"your project requirement\"")


if __name__ == "__main__":
    cli()

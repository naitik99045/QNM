"""
QNM Terminal CLI & Interactive REPL.
Provides instant command-line access to the Dual-Bottleneck Pipeline,
Pre-Installed Autonomous AI Agents, Persistent Daemon, and Token Telemetry.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
import httpx

from qnm.pipeline.orchestrator import QNMOrchestrator
from qnm.agents.code_agent import CodeAgent
from qnm.agents.research_agent import ResearchAgent
from qnm.agents.shell_agent import ShellAgent
from qnm.agents.custom_agent import resolve_agent, get_available_agents
from qnm.settings import get_settings

# Ensure Windows UTF-8 stdout/stderr compatibility and enable Virtual Terminal Processing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            h = kernel32.GetStdHandle(handle_id)
            if h and h != -1:
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass

app = typer.Typer(
    name="qnm",
    help="QNM: Quantized-Concept Network Model Engine & Autonomous Agent Suite",
    no_args_is_help=True
)
console = Console()


def _get_daemon_url() -> str:
    settings = get_settings()
    return f"http://{settings.server.host}:{settings.server.port}"


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on")
):
    """Start the persistent QNM daemon and OpenAI-compatible proxy server."""
    import uvicorn
    console.print(Panel.fit(
        f"[bold cyan]QNM Engine Daemon[/bold cyan]\n"
        f"Binding: [green]http://{host}:{port}[/green]\n"
        f"OpenAI Proxy: [green]http://{host}:{port}/v1/chat/completions[/green]\n"
        f"Native Endpoint: [green]http://{host}:{port}/qnm/query[/green]",
        title="[bold green][QNM Daemon][/bold green]"
    ))
    uvicorn.run("qnm.daemon.app:app", host=host, port=port, reload=False)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="User prompt or instruction"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Optional path to attach a file"),
    session: str = typer.Option("default", "--session", "-s", help="Conversation session ID"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Target LM Studio model ID (default: auto)")
):
    """Execute a query through the 5-stage QNM pipeline with live telemetry."""
    file_attachments = [file] if file else None
    daemon_url = _get_daemon_url()
    used_daemon = False
    result_data = None

    # Try connecting to background daemon first for < 15ms overhead
    try:
        with httpx.Client(timeout=60.0) as client:
            res = client.post(f"{daemon_url}/qnm/query", json={
                "prompt": prompt,
                "session_id": session,
                "file_attachments": file_attachments,
                "model": model
            })
            if res.status_code == 200:
                result_data = res.json()
                used_daemon = True
    except Exception:
        pass

    # If daemon is not running, run pipeline directly in-process
    if not used_daemon:
        orchestrator = QNMOrchestrator()
        result = orchestrator.process_query(
            prompt=prompt,
            session_id=session,
            file_attachments=file_attachments,
            model=model
        )
        result_data = result.model_dump()

    # Render response
    response_text = result_data["expanded_response"]
    console.print("\n[bold green]Response:[/bold green]")
    console.print(Markdown(response_text))

    # Render Telemetry Box
    tel = result_data["telemetry"]
    mode_tag = "[cyan]Daemon[/cyan]" if used_daemon else "[yellow]In-Process[/yellow]"
    telemetry_table = Table(title=f"[QNM Efficiency Telemetry] ({mode_tag})", show_header=True)
    telemetry_table.add_column("Metric", style="dim")
    telemetry_table.add_column("Value", style="bold green")

    telemetry_table.add_row("Active Model", str(result_data.get("model_used") or "auto-detected"))
    telemetry_table.add_row("Context Compression", f"{tel['context_saved_percent']}% saved ({tel['raw_context_tokens']} -> {tel['compressed_input_tokens']} tokens)")
    telemetry_table.add_row("Reasoning Generation", f"{tel['generation_saved_percent']}% saved ({tel['final_output_tokens']} -> {tel['skeleton_output_tokens']} tokens)")
    telemetry_table.add_row("Symbols Protected", str(result_data["symbols_mapped"]))
    telemetry_table.add_row("Total Latency", f"{tel['total_latency_ms']} ms")

    console.print(telemetry_table)


@app.command()
def agent(
    specialist: str = typer.Argument(..., help="Agent name: code, research, shell, or custom preset (e.g. writer, debugger)"),
    task: str = typer.Argument(..., help="Task description or goal"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Target LM Studio model ID (default: auto)")
):
    """Run a pre-installed or custom specialist autonomous agent."""
    try:
        ag = resolve_agent(specialist, model=model)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    model_display = model or "auto-detected"
    console.print(Panel.fit(
        f"Agent: [bold cyan]{ag.name}[/bold cyan]\n"
        f"Role: [dim]{ag.role_description}[/dim]\n"
        f"Model: [bold green]{model_display}[/bold green]\n"
        f"Task: [bold yellow]{task}[/bold yellow]",
        title="[Autonomous Agent]"
    ))

    result = ag.run(task, model=model)

    console.print("\n[bold green]Execution Result:[/bold green]")
    console.print(Markdown(result["final_answer"]))

    if result["trace"]:
        trace_table = Table(title="Agent Execution Trace", show_header=True)
        trace_table.add_column("Step", style="dim", width=6)
        trace_table.add_column("Action / Observation", style="cyan")
        for step in result["trace"]:
            step_num = str(step.get("iteration", "-"))
            desc = step.get("action") or step.get("thought", "")[:120]
            trace_table.add_row(step_num, desc)
        console.print(trace_table)


@app.command()
def agents():
    """List all available agents (built-in and custom YAML presets)."""
    catalog = get_available_agents()
    table = Table(title="[Catalog of Available QNM Agents]", show_header=True)
    table.add_column("Agent ID", style="bold cyan")
    table.add_column("Type", style="dim")
    table.add_column("Role", style="bold white")
    table.add_column("Enabled Tools", style="green")

    for key, spec in sorted(catalog.items()):
        tools_str = ", ".join(spec.get("tools", []))
        table.add_row(key, spec.get("type", "custom"), spec.get("role", ""), tools_str)

    console.print(table)


@app.command()
def chat(
    session: str = typer.Option("interactive_chat", "--session", "-s", help="Session ID"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Target LM Studio model ID (default: auto)")
):
    """Launch an interactive multi-turn terminal chat REPL."""
    model_tag = model or "auto-detected"
    console.print(Panel.fit(
        f"[bold cyan]QNM Interactive Concept Chat[/bold cyan]\n"
        f"Session: [bold green]{session}[/bold green]\n"
        f"Model: [bold green]{model_tag}[/bold green]\n"
        f"Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to stop.",
        title="[Interactive REPL]"
    ))

    orchestrator = QNMOrchestrator()
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            if not user_input or user_input.strip().lower() in ["exit", "quit"]:
                console.print("[dim]Goodbye![/dim]")
                break

            result = orchestrator.process_query(prompt=user_input, session_id=session, model=model)
            console.print("\n[bold green]QNM:[/bold green]")
            console.print(Markdown(result.expanded_response))
            console.print(f"[dim]{result.telemetry.summary_string()}[/dim]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]")
            break


@app.command()
def stats():
    """Display lifetime token compression and savings statistics."""
    orchestrator = QNMOrchestrator()
    data = orchestrator.store.get_telemetry_stats()

    table = Table(title="[Lifetime QNM Efficiency Scoreboard]", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold green")

    table.add_row("Total Pipeline Requests", str(data["total_requests"]))
    table.add_row("Raw Context Ingested", f"{data['total_raw_tokens']:,} tokens")
    table.add_row("Compressed Context Used", f"{data['total_compressed_tokens']:,} tokens")
    table.add_row("Context Memory Saved", f"[bold green]{data['context_saved_percent']}%[/bold green] ({data['context_saved_tokens']:,} tokens)")
    table.add_row("Reasoning Tokens Saved", f"[bold green]{data['reasoning_saved_percent']}%[/bold green] ({data['reasoning_saved_tokens']:,} tokens)")
    table.add_row("Average Latency", f"{data['avg_latency_ms']} ms")

    console.print(table)


@app.command()
def clear(
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Optional specific session ID to wipe")
):
    """Clear stored concept codes and history turns."""
    orchestrator = QNMOrchestrator()
    orchestrator.store.clear_session(session)
    target = f"session '{session}'" if session else "all sessions"
    console.print(f"[bold green][OK][/bold green] Successfully cleared concept memory for {target}.")


@app.command("lms-status")
def lms_status():
    """Check LM Studio daemon status and loaded models."""
    settings = get_settings()
    provider = orchestrator = QNMOrchestrator().provider
    is_up = provider.is_available() if hasattr(provider, "is_available") else False
    models = provider.list_models() if hasattr(provider, "list_models") else []

    table = Table(title="LM Studio Local Runtime Status", show_header=True)
    table.add_column("Component", style="dim")
    table.add_column("Status / Details", style="bold cyan")

    table.add_row("Endpoint", settings.lmstudio.base_url)
    table.add_row("CLI Path", settings.find_lms_cli() or "Not found")
    table.add_row("Server Online", "[bold green]YES[/bold green]" if is_up else "[bold red]NO (Run 'lms server start')[/bold red]")
    table.add_row("Loaded Models", ", ".join(models) if models else "None detected")

    console.print(table)


if __name__ == "__main__":
    app()

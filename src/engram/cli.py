"""engram command line interface."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from engram.brain import Brain
from engram.config import Config, load_config, save_config
from engram.embedding import EmbedError

app = typer.Typer(
    name="engram",
    help="Your second brain. Fully local, zero cloud, no PyTorch.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True, style="red")


def _brain(name: str) -> Brain:
    return Brain(name=name, cfg=load_config())


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    brain: str = typer.Option("default", "--brain", "-b", help="Brain name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
):
    """Index documents into a brain."""
    p = Path(path).expanduser()
    if not p.exists():
        err.print(f"[red]not found:[/red] {p}")
        raise typer.Exit(1)

    count = len(list(p.rglob("*"))) if p.is_dir() else 1
    if not yes and p.is_dir() and count > 200:
        typer.confirm(f"{p} has {count} entries. Ingest anyway?", abort=True)

    b = _brain(brain)
    with console.status("[bold cyan]reading + embedding…", spinner="dots"):
        try:
            res = b.ingest(p)
        except EmbedError as e:
            err.print(f"[red]{e}[/red]")
            raise typer.Exit(2)

    console.print(Panel(
        f"[green]{res['files']}[/green] files · [cyan]{res['chunks']}[/cyan] chunks · "
        f"{res['seconds']}s\n[dim]{res['skipped']} skipped[/dim]",
        title=f"ingested → {brain}", border_style="green"))
    for src, e in res["errors"]:
        err.print(f"  [yellow]![/yellow] {Path(src).name}: {e}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="What do you want to remember?"),
    brain: str = typer.Option("default", "--brain", "-b"),
    top_k: int = typer.Option(6, "--top-k", "-k"),
    answer: bool = typer.Option(False, "--answer", "-a", help="Generate an answer with a local LLM"),
):
    """Ask your notes a question."""
    b = _brain(brain)
    if len(b) == 0:
        err.print("this brain is empty — run `engram ingest <path>` first")
        raise typer.Exit(1)
    with console.status("[cyan]searching…", spinner="dots"):
        try:
            res = b.ask(question, top_k=top_k, with_answer=answer)
        except EmbedError as e:
            err.print(f"[red]{e}[/red]")
            raise typer.Exit(2)

    if res.get("answer"):
        console.print(Panel(Markdown(res["answer"]), title="answer", border_style="cyan"))
    if res.get("answer_error"):
        err.print(f"[yellow]generation failed:[/yellow] {res['answer_error']}")

    for i, h in enumerate(res["hits"], 1):
        name = Path(h["source"]).name
        console.print(f"\n[dim]──[/dim] [bold cyan]{i}.[/bold cyan] "
                      f"[green]{name}[/green] [dim]({h['score']})[/dim]")
        body = h["text"].strip()
        console.print(body[:600] + ("…" if len(body) > 600 else ""))


@app.command("ls")
def ls(brain: str = typer.Option("default", "--brain", "-b")):
    """List indexed sources."""
    b = _brain(brain)
    rows = b.sources()
    if not rows:
        console.print("[dim]empty brain[/dim]")
        return
    t = Table(title=f"{brain} — {len(b)} chunks", border_style="cyan")
    t.add_column("chunks", justify="right", style="cyan")
    t.add_column("source", style="green")
    for src, n in rows:
        t.add_row(str(n), src)
    console.print(t)


@app.command()
def stats(brain: str = typer.Option("default", "--brain", "-b")):
    """Show index stats."""
    b = _brain(brain)
    s = b.stats()
    console.print(Panel(
        "\n".join(f"[cyan]{k}[/cyan]: {v}" for k, v in s.items()),
        title=f"brain: {brain}", border_style="cyan"))


@app.command()
def config(
    model: str | None = typer.Option(None, help="Embedding model repo id"),
    llm_endpoint: str | None = typer.Option(None, help="e.g. http://localhost:11434/v1"),
    llm_model: str | None = typer.Option(None, help="e.g. qwen2.5:7b"),
    show: bool = typer.Option(False, "--show", help="Print current config"),
):
    """View or change settings."""
    cfg = load_config()
    if model:
        cfg.model = model
    if llm_endpoint:
        cfg.llm_endpoint = llm_endpoint
    if llm_model:
        cfg.llm_model = llm_model
    if model or llm_endpoint or llm_model:
        save_config(cfg)
        console.print(f"[green]saved[/green] → {cfg.path()}")
    if show or not (model or llm_endpoint or llm_model):
        for k, v in vars(cfg).items():
            console.print(f"[cyan]{k}[/cyan] = {v}")


@app.command()
def reset(brain: str = typer.Option("default", "--brain", "-b"),
          yes: bool = typer.Option(False, "--yes", "-y")):
    """Delete a brain's index."""
    if not yes:
        typer.confirm(f"Delete brain '{brain}'?", abort=True)
    _brain(brain).reset()
    console.print(f"[green]reset[/green] {brain}")


@app.command()
def warm():
    """Download and cache the embedding model (run once, offline forever after)."""
    from engram.embedding import Embedder
    try:
        with console.status("[cyan]fetching model…", spinner="dots"):
            Embedder(load_config()).load()
        console.print("[green]model ready — you can now go offline[/green]")
    except EmbedError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()

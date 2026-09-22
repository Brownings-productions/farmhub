"""Command line entry point: ``farmhub`` and ``python -m farmhub``."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from farmhub import __version__
from farmhub.core.config import Settings, load_settings
from farmhub.core.errors import ConfigError

app = typer.Typer(
    name="farmhub",
    help="FarmHub: local-first AI hub for a Norwegian homestead.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

EXIT_INVALID_CONFIG = 2


@dataclass(frozen=True)
class CliOptions:
    """Global options; the CLI layer of the config precedence (SPEC §7)."""

    config: Path | None
    dry_run: bool | None
    log_level: str | None

    def overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.dry_run is not None:
            out["dry_run"] = self.dry_run
        if self.log_level is not None:
            out["logging"] = {"level": self.log_level.upper()}
        return out


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"farmhub {__version__}")
        raise typer.Exit()


@app.callback()
def global_options(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show version."),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="TOML config file (default: config/farmhub.toml)."),
    ] = None,
    dry_run: Annotated[
        bool | None,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Override dry-run. Default is ON; production config turns it off explicitly.",
        ),
    ] = None,
    log_level: Annotated[str | None, typer.Option("--log-level", help="Log level.")] = None,
) -> None:
    ctx.obj = CliOptions(config, dry_run, log_level)


def load_cli_settings(options: CliOptions) -> Settings:
    """Load settings, or print the problem and exit non-zero: invalid config never limps on."""
    try:
        return load_settings(options.config, options.overrides())
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_INVALID_CONFIG) from exc


def _format(value: object) -> str:
    """Booleans as TOML writes them; everything else as itself."""
    return str(value).lower() if isinstance(value, bool) else str(value)


@config_app.command("check")
def config_check(ctx: typer.Context) -> None:
    """Validate configuration and print the effective settings, safety ones first."""
    options: CliOptions = ctx.obj
    settings = load_cli_settings(options)
    typer.echo("configuration OK")
    for key, value in settings.safety_summary().items():
        typer.echo(f"safety.{key} = {_format(value)}")
    typer.echo(f"logging.level = {settings.logging.level}")
    typer.echo(f"logging.file = {settings.logging.file}")

    profile = settings.active_profile()
    if profile is not None:
        typer.echo(f"profile.repo_id = {profile.repo_id}")
        typer.echo(f"profile.revision = {profile.revision}")
        typer.echo(f"profile.kv_cache_dtype = {profile.kv_cache_dtype}")
        typer.echo(f"profile.measured_by = {profile.measured_by}")

    for event, detail in settings.safety_warnings():
        typer.echo(f"WARNING: {event}: {detail}", err=True)


@app.command("serve")
def serve(ctx: typer.Context) -> None:
    """Run the HTTP server the Home Assistant conversation agent calls.

    Binds to the configured address, which defaults to loopback; a non-loopback bind
    is logged as a warning at startup (SPEC §3.6). Starting with no API token
    configured is an error, not a warning: the token is what authenticates HA.

    The LLM backend need not be running. The llm module starts degraded and is
    retried, so the server comes up either way and says so in its health.
    """
    import uvicorn

    from farmhub.api.server import create_app

    options: CliOptions = ctx.obj
    settings = load_cli_settings(options)
    try:
        api = create_app(settings)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_INVALID_CONFIG) from exc

    uvicorn.run(
        api,
        host=settings.api.bind_host,
        port=settings.api.port,
        # structlog already owns the log configuration; uvicorn's own would double
        # every line and undo the JSON formatting.
        log_config=None,
    )


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()

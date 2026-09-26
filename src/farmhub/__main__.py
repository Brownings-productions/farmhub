"""Command line entry point: ``farmhub`` and ``python -m farmhub``."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from farmhub import __version__
from farmhub.core.config import Settings, load_settings
from farmhub.core.errors import ConfigError
from farmhub.core.serving import (
    DEFAULT_ENV_PATH,
    OWNED_KEYS,
    parse_env_file,
    render_env_file,
    verify_settings,
)

app = typer.Typer(
    name="farmhub",
    help="FarmHub: local-first AI hub for a Norwegian homestead.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")
vllm_app = typer.Typer(help="The vLLM server's environment.", no_args_is_help=True)
app.add_typer(vllm_app, name="vllm")

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

    # The serving environment is part of the configuration: a profile that describes a
    # different setup than the one vLLM will be started with makes its measured numbers
    # a fiction (core/serving.py).
    try:
        note = verify_settings(settings)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_INVALID_CONFIG) from exc
    typer.echo(f"serving_env = {note if note else 'matches the active profile'}")

    for event, detail in settings.safety_warnings():
        typer.echo(f"WARNING: {event}: {detail}", err=True)


@vllm_app.command("env")
def vllm_env(
    ctx: typer.Context,
    profile_name: Annotated[
        str | None,
        typer.Option("--profile", help="Profile to render. Default: the configured one."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option("--write", help="Write the file instead of printing it."),
    ] = False,
) -> None:
    """Render the model half of the vLLM environment from a measured profile.

    The operator's own keys — image tag, cache directory, port, bind address, the WSL2
    pin-memory flag — are read from the existing file and kept. Everything the profile
    determines is replaced, so switching profiles never means editing the file by hand
    and never leaves a flag behind from the last model that was served.
    """
    options: CliOptions = ctx.obj
    settings = load_cli_settings(options)

    name = profile_name or settings.llm.profile
    if name is None:
        typer.echo(
            "error: no profile given and llm.profile is unset. A serving environment "
            "can only be rendered from a measured profile (SPEC §2).",
            err=True,
        )
        raise typer.Exit(EXIT_INVALID_CONFIG)
    profile = settings.profiles.get(name)
    if profile is None:
        known = ", ".join(sorted(settings.profiles)) or "none"
        typer.echo(f"error: no [profiles.{name}] block (known: {known})", err=True)
        raise typer.Exit(EXIT_INVALID_CONFIG)

    path = settings.llm.serving_env_file or DEFAULT_ENV_PATH
    existing = parse_env_file(path) if path.is_file() else {}
    text = render_env_file(profile, settings.llm.model, existing)

    if not write:
        typer.echo(text, nl=False)
        return
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"error: cannot write {path}: {exc}", err=True)
        raise typer.Exit(EXIT_INVALID_CONFIG) from exc
    # Never echo the contents: this file sits beside secrets.
    typer.echo(f"wrote {path} from profile {name!r} ({len(OWNED_KEYS)} keys)")


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

"""Oscar ARO per-task effort delegation provider plugin."""

from .provider import OscarAroEffortProvider


def register(ctx) -> None:
    """Register through Hermes' public PluginContext provider seam."""
    ctx.register_delegation_provider(OscarAroEffortProvider())

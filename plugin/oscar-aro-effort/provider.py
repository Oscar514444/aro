"""Minimal native route-only provider for per-task reasoning effort."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from agent.delegation_provider import DelegationProvider, DelegationRouteError


class OscarAroEffortProvider(DelegationProvider):
    """Expose a primary-selected effort on each native delegated task."""

    _EFFORTS = ("low", "medium", "high", "xhigh")


    @property
    def name(self) -> str:
        return "oscar_aro"

    @property
    def display_name(self) -> str:
        return "Oscar ARO"

    @property
    def summary(self) -> str:
        return "Preserve the configured native child route; select reasoning effort per task."

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            "reasoning_effort": {
                "type": "string",
                "enum": list(self._EFFORTS),
                "description": "Primary-selected effort for this task; omission uses fixed xhigh fallback.",
            }
        }

    def description(self, config: Mapping[str, Any]) -> str:
        return (
            "Primary selects effort per task: low for mechanical/simple work, medium for routine work, "
            "high for complex multi-step work, xhigh for the most demanding work. Omission uses fixed "
            "xhigh fallback; this is not adaptive task scoring."
        )

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        """Return only this task's effort, without changing shared route configuration."""
        if not isinstance(config, Mapping):
            raise DelegationRouteError("oscar_aro configuration must be a mapping")

        backends = config.get("backends", {})
        if backends is None:
            backends = {}
        if not isinstance(backends, Mapping):
            raise DelegationRouteError("delegation.backends must be a mapping")
        backend_settings = backends.get(self.name, {})
        if backend_settings is None:
            backend_settings = {}
        if not isinstance(backend_settings, Mapping) or backend_settings:
            raise DelegationRouteError(
                "oscar_aro accepts no backend settings; it only routes task-local reasoning_effort"
            )

        configured_fallback = config.get("reasoning_effort", "xhigh")
        if configured_fallback != "xhigh":
            raise DelegationRouteError(
                "the fixed omitted-field fallback is xhigh; configure effort on each task instead"
            )

        if "reasoning_effort" not in task:
            effort = "xhigh"
            label = "reasoning_effort=xhigh (fallback: omitted)"
        else:
            effort = task["reasoning_effort"]
            if not isinstance(effort, str) or effort not in self._EFFORTS:
                allowed = ", ".join(self._EFFORTS)
                raise DelegationRouteError(f"reasoning_effort must be one of: {allowed}")
            label = f"reasoning_effort={effort}"

        # A provider's empty route inherits PRIMARY, NOT delegation.model/provider.
        # Preserve the existing configured child route exactly, changing effort only.
        keys = ("model", "provider", "base_url", "api_key", "api_mode",
                "request_overrides", "fallback_providers")
        route = {key: deepcopy(config[key]) for key in keys if key in config}
        route.update(reasoning_effort=effort, label=label, service_tier="normal")
        return route

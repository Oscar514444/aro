"""Offline regression tests for the public ARO provider contract.

These tests supply a deliberately narrow stand-in for
agent.delegation_provider. They do not import or test Hermes itself.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any


class DelegationRouteError(Exception):
    """Narrow stand-in for the exception used by the provider contract."""


class DelegationProvider:
    """Narrow stand-in: only the superclass symbol needed at import time."""


# Do not require an installed Hermes package. The stub is intentionally limited
# to the two symbols imported by this plugin and is not an integration harness.
agent_module = types.ModuleType("agent")
agent_module.__path__ = []  # type: ignore[attr-defined]
contract_module = types.ModuleType("agent.delegation_provider")
contract_module.DelegationProvider = DelegationProvider
contract_module.DelegationRouteError = DelegationRouteError
sys.modules.setdefault("agent", agent_module)
sys.modules["agent.delegation_provider"] = contract_module

DEFAULT_PROVIDER = (
    Path(__file__).resolve().parents[1]
    / "plugin" / "oscar-aro-effort" / "provider.py"
)
PROVIDER_PATH = Path(os.environ.get("ARO_PROVIDER_PATH", DEFAULT_PROVIDER)).resolve()


def load_provider():
    if not PROVIDER_PATH.is_file():
        raise FileNotFoundError(f"provider.py not found: {PROVIDER_PATH}")
    spec = importlib.util.spec_from_file_location("aro_provider_under_test", PROVIDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load provider: {PROVIDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProviderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_provider()
        cls.provider = cls.module.OscarAroEffortProvider()

    def test_identity_and_advertised_task_parameter(self):
        self.assertEqual(self.provider.name, "oscar_aro")
        params = self.provider.task_parameters({})
        self.assertEqual(params["reasoning_effort"]["enum"], ["low", "medium", "high", "xhigh"])

    def test_each_supported_effort_is_task_local(self):
        for effort in ("low", "medium", "high", "xhigh"):
            with self.subTest(effort=effort):
                route = self.provider.route_task({"reasoning_effort": effort}, {})
                self.assertEqual(route["reasoning_effort"], effort)
                self.assertEqual(route["service_tier"], "normal")
                self.assertEqual(route["label"], f"reasoning_effort={effort}")

    def test_omission_uses_fixed_xhigh_fallback(self):
        route = self.provider.route_task({}, {})
        self.assertEqual(route["reasoning_effort"], "xhigh")
        self.assertIn("fallback: omitted", route["label"])

    def test_rejects_invalid_task_efforts(self):
        for effort in ("", "XHIGH", "auto", None, 3, True, [], {}):
            with self.subTest(effort=effort):
                with self.assertRaises(DelegationRouteError):
                    self.provider.route_task({"reasoning_effort": effort}, {})

    def test_rejects_non_mapping_config(self):
        for config in (None, [], "config"):
            with self.subTest(config=config):
                with self.assertRaises(DelegationRouteError):
                    self.provider.route_task({}, config)  # type: ignore[arg-type]

    def test_rejects_invalid_backend_and_non_fixed_fallback_settings(self):
        bad_configs = (
            {"backends": []},
            {"backends": {"oscar_aro": []}},
            {"backends": {"oscar_aro": {"model": "synthetic"}}},
            {"reasoning_effort": "high"},
            {"reasoning_effort": None},
        )
        for config in bad_configs:
            with self.subTest(config=config):
                with self.assertRaises(DelegationRouteError):
                    self.provider.route_task({}, config)

    def test_none_backend_sections_are_treated_as_empty(self):
        self.assertEqual(
            self.provider.route_task({}, {"backends": None})["reasoning_effort"], "xhigh"
        )
        self.assertEqual(
            self.provider.route_task({}, {"backends": {"oscar_aro": None}})["reasoning_effort"], "xhigh"
        )

    def test_preserves_route_deeply_and_does_not_mutate_inputs(self):
        config: dict[str, Any] = {
            "model": "synthetic-model",
            "provider": "synthetic-provider",
            "base_url": "https://example.invalid/api",
            "api_key": "synthetic-not-a-secret",
            "api_mode": "responses",
            "request_overrides": {"nested": {"items": [1, 2]}},
            "fallback_providers": [{"provider": "synthetic-fallback", "options": {"x": 1}}],
            "unrelated": {"keep": True},
        }
        task = {"reasoning_effort": "medium", "payload": {"nested": ["original"]}}
        before_config, before_task = deepcopy(config), deepcopy(task)
        route = self.provider.route_task(task, config)
        self.assertEqual(config, before_config)
        self.assertEqual(task, before_task)
        for key in ("model", "provider", "base_url", "api_key", "api_mode", "request_overrides", "fallback_providers"):
            self.assertEqual(route[key], config[key])
        for key in ("request_overrides", "fallback_providers"):
            self.assertIsNot(route[key], config[key])
        route["request_overrides"]["nested"]["items"].append(3)
        route["fallback_providers"][0]["options"]["x"] = 9
        self.assertEqual(config, before_config)


if __name__ == "__main__":
    unittest.main(verbosity=2)

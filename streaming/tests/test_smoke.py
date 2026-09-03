"""Smoke tests for the streaming service.

These assert the service imports, builds its configuration from defaults and
exposes its routes. That catches broken imports, missing dependencies and
malformed config defaults - the class of failure that went unnoticed while CI
was unable to run at all.
"""


def test_config_builds_from_defaults():
    from src.config import get_config

    config = get_config()
    assert config is not None


def test_app_exposes_health_route():
    from src.main import app

    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/health" in paths

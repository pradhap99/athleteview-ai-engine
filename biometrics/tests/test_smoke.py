"""Smoke tests for the biometrics service.

These assert the service imports, builds its settings from defaults and
exposes its routes. That catches broken imports, missing dependencies and
malformed config defaults - the class of failure that went unnoticed while CI
was unable to run at all.
"""


def test_settings_build_from_defaults():
    from src.config import get_settings

    settings = get_settings()
    assert settings is not None


def test_app_exposes_health_route():
    from src.main import app

    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/health" in paths

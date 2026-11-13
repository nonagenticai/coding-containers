"""Tests for settings module."""

import pytest

import src.config.settings as settings_mod
from src.config.settings import Settings, get_settings, reload_settings


@pytest.fixture(autouse=True)
def _reset_settings_singleton(monkeypatch):
    """Reset the settings singleton between tests.

    Also forces ENVIRONMENT=test so the production validator (which now
    requires main_api_endpoint to be supplied via env) does not fire.
    """
    monkeypatch.setenv("CODING_CONTAINERS_ENVIRONMENT", "test")
    settings_mod._settings_instance = None
    yield
    settings_mod._settings_instance = None


class TestSettingsDefaults:
    """Verify default values are correct."""

    def test_docker_base_port_default(self):
        s = Settings()
        assert s.docker_base_port == 8700

    def test_temp_dir_default(self):
        s = Settings()
        assert s.temp_dir == "/tmp"

    def test_main_api_endpoint_default(self):
        s = Settings()
        # No hardcoded fallback — env-only in production. Default empty string.
        assert s.main_api_endpoint == ""

    def test_main_api_endpoint_required_in_production(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_ENVIRONMENT", "production")
        monkeypatch.delenv("CODING_CONTAINERS_MAIN_API_ENDPOINT", raising=False)
        with pytest.raises(RuntimeError, match="main_api_endpoint"):
            Settings()

    def test_coding_github_url_default(self):
        s = Settings()
        # Env-only (config-externalization policy) — no in-code fallback.
        assert s.coding_github_url == ""

    def test_coding_github_url_required_in_production(self, monkeypatch):
        """A forgotten gitops env var must crash boot, not 500 on first provision."""
        monkeypatch.setenv("CODING_CONTAINERS_ENVIRONMENT", "production")
        monkeypatch.setenv("CODING_CONTAINERS_MAIN_API_ENDPOINT", "http://gateway:8687")
        monkeypatch.delenv("CODING_GITHUB_URL", raising=False)
        monkeypatch.delenv("CODING_CONTAINERS_CODING_GITHUB_URL", raising=False)
        with pytest.raises(RuntimeError, match="coding_github_url"):
            Settings()

    def test_coding_github_url_from_unprefixed_env(self, monkeypatch):
        monkeypatch.setenv("CODING_GITHUB_URL", "http://coding-github:8692")
        assert Settings().coding_github_url == "http://coding-github:8692"

    def test_internal_service_token_from_prefixed_env(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_INTERNAL_SERVICE_TOKEN", "shared-secret")
        s = Settings()
        assert s.internal_service_token == "shared-secret"

    def test_internal_service_token_default_empty(self):
        assert Settings().internal_service_token == ""

    def test_redis_defaults(self):
        s = Settings()
        assert s.redis.host == "localhost"
        assert s.redis.port == 6379
        assert s.redis.db == 0
        assert s.redis.password is None

    def test_worker_defaults(self):
        s = Settings()
        assert s.worker.internal_worker_port == 8700


class TestSettingsEnvOverrides:
    """Verify environment variable overrides work correctly."""

    def test_override_docker_base_port(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_DOCKER_BASE_PORT", "9100")
        s = Settings()
        assert s.docker_base_port == 9100

    def test_override_temp_dir(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_TEMP_DIR", "/var/tmp")
        s = Settings()
        assert s.temp_dir == "/var/tmp"

    def test_override_main_api_endpoint(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_MAIN_API_ENDPOINT", "http://localhost:9999")
        s = Settings()
        assert s.main_api_endpoint == "http://localhost:9999"


class TestSettingsSingleton:
    """Verify singleton/reload behavior."""

    def test_get_settings_returns_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings_creates_new_instance(self):
        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2

    def test_reload_settings_replaces_singleton(self):
        get_settings()
        reloaded = reload_settings()
        current = get_settings()
        assert current is reloaded


class TestDatabaseManager:
    """Verify the database manager stub behavior."""

    def test_get_database_manager_returns_stub(self):
        s = Settings()
        db_manager = s.get_database_manager()
        # Stub should raise NotImplementedError on any attribute access
        with pytest.raises(NotImplementedError, match="DatabaseManager is not available"):
            _ = db_manager.base_paths

    def test_get_database_manager_raises_on_any_attribute(self):
        s = Settings()
        db_manager = s.get_database_manager()
        with pytest.raises(NotImplementedError):
            _ = db_manager.some_random_attribute
        with pytest.raises(NotImplementedError):
            db_manager.connect()


class TestCodeGraphDbPath:
    """Verify code graph database path generation."""

    def test_get_code_graph_db_path_includes_graph_id(self):
        s = Settings()
        path = s.get_code_graph_db_path("test-graph")
        assert "test-graph.db" in path

    def test_get_code_graph_db_path_uses_temp_dir_by_default(self):
        s = Settings()
        path = s.get_code_graph_db_path("my-graph")
        assert path.startswith(s.temp_dir)

    def test_get_code_graph_db_path_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("CODE_GRAPH_DB_DIR", "/data/graphs")
        s = Settings()
        path = s.get_code_graph_db_path("custom-graph")
        assert path == "/data/graphs/custom-graph.db"

    def test_get_code_graph_db_path_different_ids_different_paths(self):
        s = Settings()
        p1 = s.get_code_graph_db_path("graph-a")
        p2 = s.get_code_graph_db_path("graph-b")
        assert p1 != p2
        assert "graph-a.db" in p1
        assert "graph-b.db" in p2


class TestReaperAndCapDefaults:
    """Phase C-1 knobs must default to fully inert / safe values."""

    def test_reaper_disabled_by_default(self):
        assert Settings().reaper_enabled is False

    def test_reaper_dry_run_by_default(self):
        assert Settings().reaper_dry_run is True

    def test_reaper_ttl_default_generous(self):
        assert Settings().reaper_ttl_seconds == 86400

    def test_reaper_interval_default(self):
        assert Settings().reaper_interval_seconds == 300

    def test_legacy_reap_off_by_default(self):
        s = Settings()
        assert s.reap_legacy_unlabelled is False
        assert s.reap_legacy_ttl_seconds == 604800

    def test_max_inflight_unlimited_by_default(self):
        assert Settings().max_inflight_jobs == 0

    def test_admission_reject_mode_default(self):
        s = Settings()
        assert s.admission_reject_mode == "reject"
        assert s.admission_wait_timeout_seconds == 0

    def test_env_override_enables_reaper(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_REAPER_ENABLED", "true")
        monkeypatch.setenv("CODING_CONTAINERS_MAX_INFLIGHT_JOBS", "25")
        s = Settings()
        assert s.reaper_enabled is True
        assert s.max_inflight_jobs == 25

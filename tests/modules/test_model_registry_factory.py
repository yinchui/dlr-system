from pathlib import Path

import pytest

import modules.model_registry_factory as registry_factory
from modules.model_registry import ModelKey, ModelRegistry
from modules.supabase_model_registry import SupabaseModelRegistry


SUPABASE_URL = "https://ciapxhuldarsupmvrgwu.supabase.co"


def test_factory_uses_local_registry_when_supabase_is_not_configured(tmp_path):
    registry = registry_factory.create_model_registry(
        model_dir=tmp_path / "models",
        secrets={},
        environ={},
    )

    assert type(registry) is ModelRegistry
    assert registry.model_dir == (tmp_path / "models").resolve()


def test_factory_uses_supabase_registry_with_default_private_bucket(
    tmp_path, monkeypatch
):
    captured = {}
    fake_store = object()

    def from_credentials(url, secret_key, *, bucket):
        captured.update(url=url, secret_key=secret_key, bucket=bucket)
        return fake_store

    monkeypatch.setattr(
        registry_factory.SupabaseModelStore,
        "from_credentials",
        from_credentials,
    )

    registry = registry_factory.create_model_registry(
        model_dir=tmp_path / "unused-local-models",
        secrets={
            "DLR_SUPABASE_URL": SUPABASE_URL,
            "DLR_SUPABASE_SECRET_KEY": "server-secret-value",
        },
        environ={},
    )

    assert isinstance(registry, SupabaseModelRegistry)
    assert registry.store is fake_store
    assert captured == {
        "url": SUPABASE_URL,
        "secret_key": "server-secret-value",
        "bucket": "dlr-models",
    }
    assert registry.model_dir != Path(tmp_path / "unused-local-models").resolve()
    assert "server-secret-value" not in repr(registry)


def test_factory_preserves_runtime_audit_configuration(tmp_path, monkeypatch):
    fake_store = object()
    logger = object()
    monkeypatch.setattr(
        registry_factory.SupabaseModelStore,
        "from_credentials",
        lambda *args, **kwargs: fake_store,
    )

    registry = registry_factory.create_model_registry(
        model_dir=tmp_path / "models",
        secrets={
            "DLR_SUPABASE_URL": SUPABASE_URL,
            "DLR_SUPABASE_SECRET_KEY": "server-secret-value",
        },
        environ={},
        audit_logger=logger,
        audit_run_id="run-a",
        audit_result_id="result-a",
    )

    assert registry.audit_logger is logger
    assert registry.audit_run_id == "run-a"
    assert registry.audit_result_id == "result-a"


def test_factory_never_places_supabase_secret_in_audit_events(
    tmp_path, monkeypatch
):
    secret = "server-secret-value"
    events = []

    class CapturingLogger:
        def write(self, event):
            events.append(event)
            return True

    monkeypatch.setattr(
        registry_factory.SupabaseModelStore,
        "from_credentials",
        lambda *args, **kwargs: object(),
    )
    registry = registry_factory.create_model_registry(
        model_dir=tmp_path / "models",
        secrets={
            "DLR_SUPABASE_URL": SUPABASE_URL,
            "DLR_SUPABASE_SECRET_KEY": secret,
        },
        environ={},
        audit_logger=CapturingLogger(),
        audit_run_id="run-a",
        audit_result_id="result-a",
    )

    assert registry._write_audit(
        ModelKey("project-a", "line-a", "001", "wind_speed"),
        stage="factory_test",
    ) is True
    assert events
    assert secret not in repr(events)


def test_factory_rejects_partial_configuration_without_local_fallback(tmp_path):
    with pytest.raises(ValueError, match="must be configured together"):
        registry_factory.create_model_registry(
            model_dir=tmp_path / "models",
            secrets={"DLR_SUPABASE_SECRET_KEY": "server-secret-value"},
            environ={},
        )

    assert not (tmp_path / "models").exists()


def test_streamlit_page_uses_registry_factory_without_exposing_secrets():
    app_source = (
        Path(__file__).resolve().parents[2] / "dispatch_app_st.py"
    ).read_text(encoding="utf-8")

    assert "create_model_registry(" in app_source
    assert "DLR_SUPABASE_SECRET_KEY" not in app_source
    assert "st.write(st.secrets" not in app_source
    assert "st.json(st.secrets" not in app_source

from pathlib import Path

from config import config


def test_sag_validation_defaults_are_safe():
    defaults = config.SAG_VALIDATION_DEFAULTS

    assert defaults["formula_version"] == "CN-patent-2025-v1"
    assert 0 < defaults["min_angle_deg"]
    assert defaults["max_angle_deg"] < 90
    assert defaults["gravity_m_s2"] == 9.80665
    assert defaults["recovery_samples"] >= 2


def test_runtime_directories_default_under_project_root():
    assert config.MODEL_DIR == config.PROJECT_ROOT / "models"
    assert config.AUDIT_LOG_DIR == config.PROJECT_ROOT / "runtime" / "audit"
    assert config.SAG_RESULT_DIR == config.PROJECT_ROOT / "runtime" / "sag_results"


def test_runtime_directories_can_be_overridden_by_environment(monkeypatch, tmp_path):
    runtime_paths = {
        "DLR_MODEL_DIR": (config.PROJECT_ROOT / "models", tmp_path / "models"),
        "DLR_AUDIT_LOG_DIR": (
            config.PROJECT_ROOT / "runtime" / "audit",
            tmp_path / "audit",
        ),
        "DLR_SAG_RESULT_DIR": (
            config.PROJECT_ROOT / "runtime" / "sag_results",
            tmp_path / "sag_results",
        ),
    }

    for env_name, (default, override) in runtime_paths.items():
        monkeypatch.setenv(env_name, str(override))
        assert config.resolve_runtime_path(env_name, default) == Path(override)

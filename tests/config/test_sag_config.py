from pathlib import Path

from config import config


def test_sag_validation_defaults_match_approved_plan():
    assert config.SAG_VALIDATION_DEFAULTS == {
        "formula_version": "CN-patent-2025-v1",
        "gravity_m_s2": 9.80665,
        "min_angle_deg": 0.05,
        "max_angle_deg": 89.5,
        "reference_temp_c": 20.0,
        "reference_tension_n": 20000.0,
        "span_m": 300.0,
        "elastic_modulus_pa": 7.0e10,
        "area_m2": 6.75e-4,
        "thermal_expansion_per_c": 1.9e-5,
        "base_threshold_c": 5.0,
        "recovery_ratio": 0.6,
        "recovery_samples": 3,
        "recovery_alpha": 0.25,
    }


def test_runtime_directories_default_under_project_root():
    assert config.MODEL_DIR == config.PROJECT_ROOT / "models"
    assert config.AUDIT_LOG_DIR == config.PROJECT_ROOT / "runtime" / "logs"
    assert config.SAG_RESULT_DIR == config.PROJECT_ROOT / "runtime" / "results" / "sag"


def test_runtime_directories_can_be_overridden_by_environment(monkeypatch, tmp_path):
    runtime_paths = {
        "DLR_MODEL_DIR": (config.PROJECT_ROOT / "models", tmp_path / "models"),
        "DLR_AUDIT_LOG_DIR": (
            config.PROJECT_ROOT / "runtime" / "logs",
            tmp_path / "audit",
        ),
        "DLR_SAG_RESULT_DIR": (
            config.PROJECT_ROOT / "runtime" / "results" / "sag",
            tmp_path / "sag_results",
        ),
    }

    for env_name, (default, override) in runtime_paths.items():
        monkeypatch.setenv(env_name, str(override))
        assert config.resolve_runtime_path(env_name, default) == Path(override)

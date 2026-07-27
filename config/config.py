# config/config.py
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPABASE_MODEL_SETTING_NAMES = (
    "DLR_SUPABASE_URL",
    "DLR_SUPABASE_SECRET_KEY",
    "DLR_SUPABASE_MODEL_BUCKET",
)
_SUPABASE_PROJECT_HOST = re.compile(r"^[a-z0-9]{20}\.supabase\.co$")


@dataclass(frozen=True)
class SupabaseModelConfig:
    url: str
    secret_key: str = field(repr=False)
    bucket: str = "dlr-models"


def _optional_setting(
    name: str,
    *,
    secrets: Mapping[str, object],
    environ: Mapping[str, str],
) -> Optional[str]:
    value = secrets.get(name)
    if value is None:
        value = environ.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    return normalized or None


def _validated_supabase_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ValueError("Supabase URL is invalid") from None
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or _SUPABASE_PROJECT_HOST.fullmatch(parsed.hostname) is None
    ):
        raise ValueError("Supabase URL must be a root HTTPS project URL")
    return f"https://{parsed.hostname}"


def load_supabase_model_config(
    *,
    secrets: Optional[Mapping[str, object]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[SupabaseModelConfig]:
    secret_values = secrets or {}
    environment = os.environ if environ is None else environ
    url = _optional_setting(
        "DLR_SUPABASE_URL",
        secrets=secret_values,
        environ=environment,
    )
    secret_key = _optional_setting(
        "DLR_SUPABASE_SECRET_KEY",
        secrets=secret_values,
        environ=environment,
    )
    if url is None and secret_key is None:
        return None
    if url is None or secret_key is None:
        raise ValueError(
            "DLR_SUPABASE_URL and DLR_SUPABASE_SECRET_KEY must be configured together"
        )
    bucket = _optional_setting(
        "DLR_SUPABASE_MODEL_BUCKET",
        secrets=secret_values,
        environ=environment,
    )
    return SupabaseModelConfig(
        url=_validated_supabase_url(url),
        secret_key=secret_key,
        bucket=bucket or "dlr-models",
    )


def resolve_runtime_path(env_name, default):
    """Resolve a runtime directory from an environment override or default."""
    configured_path = os.environ.get(env_name)
    return Path(configured_path).expanduser() if configured_path else Path(default)


APP_TITLE = "DLR动态增容评估系统"
DEFAULT_INTERVAL_MINUTES = 30
DEFAULT_MAX_ALLOW_TEMP = 80.0
DLR_SAFETY_FACTOR = 0.8
PROJECT_TIMEZONE = "Asia/Shanghai"

MODEL_DIR = resolve_runtime_path("DLR_MODEL_DIR", PROJECT_ROOT / "models")
AUDIT_LOG_DIR = resolve_runtime_path(
    "DLR_AUDIT_LOG_DIR", PROJECT_ROOT / "runtime" / "logs"
)
SAG_RESULT_DIR = resolve_runtime_path(
    "DLR_SAG_RESULT_DIR", PROJECT_ROOT / "runtime" / "results" / "sag"
)
DEMO_DATA_DIR = Path("assets/demo_data")

# Backend-only conservative mechanical defaults in SI units; not measured values.
STANDARD_CONDUCTORS = {
    "4×JL/G1A-630/45": {
        "D0": 0.0338,
        "R_low_25": 4.680e-5,
        "R_high_75": 5.830e-5,
        "R_high_200": 8.740e-5,
        "materials": [
            {"type": "aluminum", "density": 1.701},
            {"type": "steel", "density": 0.350},
        ],
        "area_m2": 6.75e-4,
        "elastic_modulus_pa": 7.0e10,
        "thermal_expansion_per_c": 19.6e-6,
        "rated_tensile_strength_n": 1.50e5,
        "mass_per_length_kg_m": 2.051,
    },
    "ACSR Drake (795 kcmil)": {
        "D0": 0.0281,
        "R_low_25": 7.283e-5,
        "R_high_75": 8.688e-5,
        "R_high_200": 1.220e-4,
        "materials": [
            {"type": "aluminum", "density": 1.116},
            {"type": "steel", "density": 0.5126},
        ],
        "area_m2": 4.685e-4,
        "elastic_modulus_pa": 7.0e10,
        "thermal_expansion_per_c": 19.1e-6,
        "rated_tensile_strength_n": 1.40e5,
        "mass_per_length_kg_m": 1.6286,
    },
}

PHYSICAL_BOUNDS = {
    "wind_speed": (0.0, 75.0),
    "ambient_temp": (-60.0, 70.0),
}

CORRECTION_DEFAULTS = {
    "ref_height_m": 10.0,
    "line_height_m": 20.0,
    "roughness_alpha": 0.15,
    "temp_lapse_rate": 0.0065,
    "humidity_factor": 0.95,
    "ground_albedo": 0.35,
    "line_azimuth_deg": 90.0,
}

SAG_VALIDATION_DEFAULTS = {
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

MODEL_BUNDLE_FILES = {
    "wind_speed": "wind_speed_bundle.joblib",
    "ambient_temp": "ambient_temp_bundle.joblib",
}

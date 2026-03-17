# config/config.py
from pathlib import Path

APP_TITLE = "DLR动态增容评估系统"
DEFAULT_INTERVAL_MINUTES = 30
DEFAULT_MAX_ALLOW_TEMP = 80.0
MODEL_DIR = Path("models")
DEMO_DATA_DIR = Path("assets/demo_data")

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
    },
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

MODEL_BUNDLE_FILES = {
    "wind_speed": "wind_speed_bundle.joblib",
    "ambient_temp": "ambient_temp_bundle.joblib",
}

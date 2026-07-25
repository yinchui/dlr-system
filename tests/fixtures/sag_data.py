import numpy as np
import pandas as pd


def drake_conductor():
    return {
        "D0": 0.02814,
        "R_low_25": 7.283e-5,
        "R_high_75": 8.688e-5,
        "R_high_200": 1.220e-4,
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "area_m2": 4.685e-4,
        "elastic_modulus_pa": 7.0e10,
        "thermal_expansion_per_c": 19.1e-6,
        "rated_tensile_strength_n": 1.40e5,
        "mass_per_length_kg_m": 1.6286,
        "materials": [
            {"type": "aluminum", "mass": 1.116},
            {"type": "steel", "mass": 0.5126},
        ],
    }


def make_line_data(*, times=2, include_operating_current=False):
    timestamps = pd.date_range(
        "2026-07-23 00:00",
        periods=times,
        freq="30min",
        tz="Asia/Shanghai",
    )
    line_data = {
        "positions": ["001", "002"],
        "datetimes": timestamps,
        "max_currents": np.array(
            [
                np.linspace(1000.0, 1010.0, times),
                np.linspace(950.0, 960.0, times),
            ]
        ),
        "local_temps": np.array(
            [
                np.linspace(25.0, 26.0, times),
                np.linspace(24.0, 25.0, times),
            ]
        ),
        "winds": np.full((2, times), 3.0),
        "angles": np.full((2, times), 90.0),
        "solar": np.linspace(0.0, 100.0, times),
        "elevations": np.array([1000.0, 1010.0]),
        "tower_coords": {
            "001": {"lon": 120.0, "lat": 40.0},
            "002": {"lon": 120.0035, "lat": 40.0},
        },
        "run_id": "dlr-run-1",
    }
    if include_operating_current:
        line_data["operating_currents"] = np.full((2, times), 600.0)
    return line_data


def make_inclination_row(**changes):
    values = {
        "tower_id": "001",
        "timestamp": pd.Timestamp("2026-07-23 00:00", tz="Asia/Shanghai"),
        "angle_deg": 1.0,
    }
    values.update(changes)
    return pd.Series(values)

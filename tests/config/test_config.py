from config.config import APP_TITLE, CORRECTION_DEFAULTS, STANDARD_CONDUCTORS


def test_conductor_catalog_and_defaults_are_available():
    assert APP_TITLE == "DLR动态增容评估系统"
    assert "4×JL/G1A-630/45" in STANDARD_CONDUCTORS
    assert CORRECTION_DEFAULTS["ref_height_m"] == 10.0
    assert CORRECTION_DEFAULTS["line_height_m"] == 20.0

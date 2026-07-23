from config import config


def test_conductor_catalog_and_defaults_are_available():
    assert config.APP_TITLE == "DLR动态增容评估系统"
    assert "4×JL/G1A-630/45" in config.STANDARD_CONDUCTORS
    assert config.CORRECTION_DEFAULTS["ref_height_m"] == 10.0
    assert config.CORRECTION_DEFAULTS["line_height_m"] == 20.0


def test_runtime_and_physical_defaults_are_available():
    assert config.PROJECT_TIMEZONE == "Asia/Shanghai"
    assert config.PHYSICAL_BOUNDS["wind_speed"] == (0.0, 75.0)
    assert config.PHYSICAL_BOUNDS["ambient_temp"] == (-60.0, 70.0)
    assert config.MODEL_DIR.name == "models"


def test_conductor_catalog_includes_sag_mechanical_defaults():
    required_keys = {
        "area_m2",
        "elastic_modulus_pa",
        "thermal_expansion_per_c",
        "rated_tensile_strength_n",
        "mass_per_length_kg_m",
    }

    for conductor in config.STANDARD_CONDUCTORS.values():
        assert required_keys <= conductor.keys()
        assert all(conductor[key] > 0 for key in required_keys)

    assert config.STANDARD_CONDUCTORS["ACSR Drake (795 kcmil)"]["area_m2"] > 0

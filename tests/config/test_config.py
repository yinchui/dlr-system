import pytest

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


def test_dlr_safety_factor_is_explicit_and_valid():
    assert config.DLR_SAFETY_FACTOR == 0.8
    assert 0.0 < config.DLR_SAFETY_FACTOR <= 1.0


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


def test_supabase_model_config_is_absent_when_url_and_secret_are_absent():
    assert config.load_supabase_model_config(secrets={}, environ={}) is None


def test_supabase_model_config_reads_root_secrets_and_defaults_bucket():
    secret = "server-secret-value"

    result = config.load_supabase_model_config(
        secrets={
            "DLR_SUPABASE_URL": "https://ciapxhuldarsupmvrgwu.supabase.co/",
            "DLR_SUPABASE_SECRET_KEY": secret,
        },
        environ={},
    )

    assert result.url == "https://ciapxhuldarsupmvrgwu.supabase.co"
    assert result.secret_key == secret
    assert result.bucket == "dlr-models"
    assert secret not in repr(result)


@pytest.mark.parametrize(
    "values",
    [
        {"DLR_SUPABASE_URL": "https://ciapxhuldarsupmvrgwu.supabase.co"},
        {"DLR_SUPABASE_SECRET_KEY": "server-secret-value"},
    ],
)
def test_supabase_model_config_rejects_partial_credentials(values):
    with pytest.raises(ValueError, match="must be configured together") as error:
        config.load_supabase_model_config(secrets=values, environ={})

    assert "server-secret-value" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://ciapxhuldarsupmvrgwu.supabase.co",
        "https://user:password@ciapxhuldarsupmvrgwu.supabase.co",
        "https://ciapxhuldarsupmvrgwu.supabase.co:443",
        "https://ciapxhuldarsupmvrgwu.supabase.co/rest/v1",
        "https://ciapxhuldarsupmvrgwu.supabase.co?key=value",
        "https://ciapxhuldarsupmvrgwu.supabase.co#fragment",
        "https://example.com",
        "https://bad_ref.supabase.co",
    ],
)
def test_supabase_model_config_rejects_unsafe_or_unexpected_url(url):
    secret = "server-secret-value"

    with pytest.raises(ValueError, match="Supabase URL") as error:
        config.load_supabase_model_config(
            secrets={
                "DLR_SUPABASE_URL": url,
                "DLR_SUPABASE_SECRET_KEY": secret,
            },
            environ={},
        )

    assert secret not in str(error.value)


def test_supabase_model_config_uses_environment_when_secrets_are_absent():
    result = config.load_supabase_model_config(
        secrets={},
        environ={
            "DLR_SUPABASE_URL": "https://ciapxhuldarsupmvrgwu.supabase.co",
            "DLR_SUPABASE_SECRET_KEY": "environment-secret",
            "DLR_SUPABASE_MODEL_BUCKET": "private-models",
        },
    )

    assert result.bucket == "private-models"
    assert result.secret_key == "environment-secret"

"""Official IEEE 738 Drake steady-state regression reference."""

import importlib.util
from pathlib import Path

import pytest

from modules.thermal_engine import ThermalCalculator


_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "ieee738_reference.py"
_fixture_spec = importlib.util.spec_from_file_location("ieee738_reference", _FIXTURE_PATH)
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
DRAKE_STEADY_PARAMS = _fixture_module.DRAKE_STEADY_PARAMS


def test_ieee_738_drake_steady_reference():
    """IEEE Std 738-2023 4.6.1 reproduces the published Drake heat balance."""
    result = ThermalCalculator().calculate_heat_balance(DRAKE_STEADY_PARAMS)

    assert result.q_convection_natural == pytest.approx(42.42, abs=0.15)
    assert result.q_convection_low_re == pytest.approx(82.10, abs=0.20)
    assert result.q_convection_high_re == pytest.approx(77.06, abs=0.20)
    assert result.q_convection == pytest.approx(82.10, abs=0.20)
    assert result.q_radiation == pytest.approx(39.11, abs=0.15)
    assert result.q_solar == pytest.approx(22.45, abs=0.25)
    assert result.resistance == pytest.approx(9.391e-5, rel=2e-3)
    assert result.current_a == pytest.approx(1025.0, abs=2.0)

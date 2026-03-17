import numpy as np

from modules.thermal_engine import LineAnalyzer, ThermalCalculator


def test_calculate_max_current_for_points_returns_expected_shapes():
    calculator = ThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    result = analyzer.calculate_max_current_for_points(
        observation_points=np.array([0.36]),
        elevations=np.array([1100.0]),
        temps=np.array([[20.0, 21.0]]),
        winds=np.array([[3.0, 3.2]]),
        angles=np.array([[90.0, 95.0]]),
        solar=np.array([0.0, 50.0]),
        times=np.array([0.0, 1.0]),
        max_temp=80.0,
        terrain_data={0: {"slope": 12.0, "aspect": 270.0, "elevation": 1100.0}},
    )
    assert result["max_currents"].shape == (1, 2)
    assert result["corrected_winds"][0, 0] > 0

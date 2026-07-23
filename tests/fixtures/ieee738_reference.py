"""IEEE Std 738-2023 4.6.1 Drake steady-state reference inputs."""

# IEEE Std 738-2023 4.6.1, pages 28-32: 795 kcmil 26/7 Drake ACSR.
# Temperatures are degrees C, dimensions are SI, and resistance is ohms/m.
# Equation (9)'s first printed line says 1.05, but its substitution and published
# result use the standard 1.01 coefficient. The reference implementation must use
# 1.01 so that it reproduces this official example.
DRAKE_STEADY_PARAMS = {
    "D0": 0.02814,
    "R_low_25": 7.283e-5,
    "R_high_75": 8.688e-5,
    "R_high_200": 1.220e-4,
    "emissivity": 0.8,
    "absorptivity": 0.8,
    "T_a": 40.0,
    "T_s": 100.0,
    "T_avg": 100.0,
    "wind_speed": 0.61,
    "wind_angle": 90.0,
    "elevation": 0.0,
    "latitude": 30.0,
    "line_azimuth": 90.0,
    "day_of_year": 161,
    "time": 11.0,
}

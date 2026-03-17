import pandas as pd

from modules.weather_correction import CorrectionOptions, WeatherCorrectionService


def test_weather_correction_applies_vertical_terrain_and_direction_factors():
    df = pd.DataFrame(
        {
            "position": [36],
            "ambient_temp": [20.0],
            "wind_speed": [4.0],
            "wind_direction": [10.0],
            "solar_radiation": [600.0],
            "humidity": [25.0],
        }
    )
    terrain_lookup = {36: {"slope": 20.0, "aspect": 190.0, "elevation": 1100.0}}
    corrected = WeatherCorrectionService().apply(
        df,
        terrain_lookup=terrain_lookup,
        options=CorrectionOptions(line_azimuth_deg=0.0),
    )
    assert corrected.loc[0, "wind_speed_corrected"] > corrected.loc[0, "wind_speed_raw"]
    assert corrected.loc[0, "wind_angle_factor"] == 1.15

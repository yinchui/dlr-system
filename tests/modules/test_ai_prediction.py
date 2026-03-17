import pandas as pd

from modules.ai_prediction import ModelBundle, ResidualPredictor


class OffsetModel:
    def predict(self, features):
        return [0.5] * len(features)


def test_predictor_returns_physical_plus_residual_prediction():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-10 00:00", "2025-12-10 01:00"]),
            "wind_speed_physical": [3.0, 4.0],
        }
    )
    predictor = ResidualPredictor(
        {
            "wind_speed": ModelBundle(
                target_name="wind_speed",
                feature_columns=["hour_sin", "hour_cos", "wind_speed_physical"],
                model=OffsetModel(),
            )
        }
    )
    predicted = predictor.predict(df, target_name="wind_speed", physical_col="wind_speed_physical")
    assert predicted["wind_speed_residual"].tolist() == [0.5, 0.5]
    assert predicted["wind_speed_final"].tolist() == [3.5, 4.5]

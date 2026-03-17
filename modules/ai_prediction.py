# modules/ai_prediction.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    target_name: str
    feature_columns: list
    model: object
    scaler: Optional[object] = None


class ResidualPredictor:
    def __init__(self, bundles: Optional[dict] = None):
        self.bundles = bundles or {}

    @classmethod
    def from_directory(cls, model_dir: Path):
        bundles = {}
        for bundle_path in model_dir.glob("*_bundle.joblib"):
            payload = joblib.load(bundle_path)
            bundles[payload["target_name"]] = ModelBundle(**payload)
        return cls(bundles)

    def build_features(self, df: pd.DataFrame, physical_col: str) -> pd.DataFrame:
        features = df.copy()
        features["hour"] = pd.to_datetime(features["timestamp"]).dt.hour
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_of_year"] = pd.to_datetime(features["timestamp"]).dt.dayofyear
        features["lag_1"] = features[physical_col].shift(1).bfill()
        return features

    def predict(self, df: pd.DataFrame, target_name: str, physical_col: str) -> pd.DataFrame:
        output = df.copy()
        bundle = self.bundles.get(target_name)
        if bundle is None:
            output[f"{target_name}_residual"] = 0.0
            output[f"{target_name}_final"] = output[physical_col]
            output["used_ai"] = False
            return output

        features = self.build_features(df, physical_col)
        feature_frame = features[bundle.feature_columns]
        if bundle.scaler is not None:
            feature_frame = bundle.scaler.transform(feature_frame)
        residual = bundle.model.predict(feature_frame)
        output[f"{target_name}_residual"] = residual
        output[f"{target_name}_final"] = output[physical_col] + residual
        output["used_ai"] = True
        return output

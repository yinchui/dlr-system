from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

from config.config import (
    DEFAULT_INTERVAL_MINUTES,
    PHYSICAL_BOUNDS,
    PROJECT_TIMEZONE,
)


_OPTIONAL_FEATURES = {
    "wind_direction": (("wind_direction", "wind_direction_local"), 0.0),
    "solar_radiation": (
        ("solar_radiation_local", "solar_radiation"),
        0.0,
    ),
    "humidity": (("humidity", "humidity_local"), 50.0),
    "elevation": (("elevation", "altitude"), 0.0),
    "slope": (("slope", "terrain_slope"), 0.0),
    "aspect": (("aspect", "terrain_aspect"), 0.0),
}

_TARGET_PHYSICAL_COLUMNS = {
    "wind_speed": frozenset(
        {
            "wind_speed",
            "wind_speed_physical",
            "wind_speed_raw",
            "wind_speed_local",
            "wind_speed_corrected",
            "wind_speed_terrain_corrected",
        }
    ),
    "ambient_temp": frozenset(
        {
            "ambient_temp",
            "ambient_temp_physical",
            "ambient_temp_raw",
            "ambient_temp_local",
            "ambient_temp_corrected",
            "ambient_temp_terrain_corrected",
        }
    ),
}


class FeatureBuilder:
    """Build deterministic weather features without crossing tower datasets."""

    _DERIVED_FEATURE_COLUMNS = (
        "lag_1",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "wind_direction_sin",
        "wind_direction_cos",
        "solar_radiation_feature",
        "humidity_feature",
        "elevation_feature",
        "slope_feature",
        "aspect_feature",
    )
    _DATASET_COLUMNS = (
        "source_file_hash",
        "source_file_hash_physical",
        "source_file_hash_truth",
        "dataset_id",
        "dataset_role",
    )

    def __init__(self, cadence_minutes: Real = DEFAULT_INTERVAL_MINUTES):
        if (
            isinstance(cadence_minutes, bool)
            or not isinstance(cadence_minutes, Real)
            or not np.isfinite(cadence_minutes)
            or cadence_minutes <= 0
        ):
            raise ValueError("cadence_minutes must be a positive finite number")
        try:
            cadence = pd.Timedelta(minutes=float(cadence_minutes))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "cadence_minutes must define a valid interval"
            ) from exc
        if pd.isna(cadence) or cadence <= pd.Timedelta(0):
            raise ValueError("cadence_minutes must define a valid interval")
        self.cadence_minutes = float(cadence_minutes)
        self._cadence_nanoseconds = int(cadence.value)

    def feature_columns(self, physical_col: str) -> list[str]:
        if not isinstance(physical_col, str) or not physical_col.strip():
            raise ValueError("physical_col must be a non-empty string")
        return [physical_col, *self._DERIVED_FEATURE_COLUMNS]

    @staticmethod
    def _numeric_column(
        frame: pd.DataFrame,
        column: str,
        *,
        label: str,
    ) -> pd.Series:
        try:
            numeric = pd.to_numeric(frame[column], errors="coerce").astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must contain finite numeric values") from exc
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} must contain finite numeric values")
        return numeric

    @staticmethod
    def _identity_values(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            raise ValueError(f"missing required column: {column}")
        if frame[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
        values = frame[column].map(lambda value: str(value).strip())
        if values.eq("").any():
            raise ValueError(f"{column} cannot contain empty values")
        return values

    @staticmethod
    def _timestamps(frame: pd.DataFrame) -> pd.Series:
        if "timestamp" not in frame.columns:
            raise ValueError("missing required column: timestamp")
        try:
            timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
            if timestamps.isna().any():
                raise ValueError("timestamp contains missing or invalid values")
            if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
                return timestamps.dt.tz_convert(PROJECT_TIMEZONE)
            if pd.api.types.is_datetime64_dtype(timestamps.dtype):
                return timestamps.dt.tz_localize(
                    PROJECT_TIMEZONE,
                    ambiguous="raise",
                    nonexistent="raise",
                )

            normalized = []
            for value in frame["timestamp"]:
                timestamp = pd.Timestamp(value)
                if pd.isna(timestamp):
                    raise ValueError(
                        "timestamp contains missing or invalid values"
                    )
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(
                        PROJECT_TIMEZONE,
                        ambiguous="raise",
                        nonexistent="raise",
                    )
                else:
                    timestamp = timestamp.tz_convert(PROJECT_TIMEZONE)
                normalized.append(timestamp)
            return pd.Series(
                pd.DatetimeIndex(normalized),
                index=frame.index,
                name="timestamp",
            )
        except (
            AmbiguousTimeError,
            NonExistentTimeError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ValueError("timestamp must contain valid datetime values") from exc

    @classmethod
    def _group_columns(cls, frame: pd.DataFrame) -> list[str]:
        columns = []
        if "line_id" in frame.columns:
            columns.append("line_id")
        columns.append("tower_id")
        columns.extend(
            column for column in cls._DATASET_COLUMNS if column in frame.columns
        )
        return columns

    def _ordered_with_segments(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy(deep=True)
        working["__feature_row_order__"] = np.arange(len(working), dtype=np.int64)
        group_columns = self._group_columns(working)
        duplicate_columns = [*group_columns, "timestamp"]
        if working.duplicated(subset=duplicate_columns, keep=False).any():
            raise ValueError(
                "duplicate tower timestamp in one dataset is not allowed"
            )

        ordered = working.sort_values(
            [*group_columns, "timestamp", "__feature_row_order__"],
            kind="mergesort",
        )
        segments = np.empty(len(ordered), dtype=np.int64)
        next_segment = 0
        grouped = ordered.groupby(group_columns, sort=False, dropna=False)
        for _, positions in grouped.indices.items():
            positions = np.asarray(positions, dtype=np.int64)
            group_timestamps = ordered.iloc[positions]["timestamp"]
            group_segments = np.zeros(len(positions), dtype=np.int64)
            if len(positions) > 1:
                differences = np.diff(
                    group_timestamps.astype("int64").to_numpy()
                )
                breaks = differences != self._cadence_nanoseconds
                group_segments[1:] = np.cumsum(breaks)
            group_segments += next_segment
            segments[positions] = group_segments
            next_segment = int(group_segments.max()) + 1
        ordered["__continuous_segment__"] = segments
        return ordered

    def continuous_segments(self, frame: pd.DataFrame) -> pd.Series:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        working = frame.copy(deep=True)
        working["tower_id"] = self._identity_values(working, "tower_id")
        if "line_id" in working.columns:
            working["line_id"] = self._identity_values(working, "line_id")
        working["timestamp"] = self._timestamps(working)
        for column in self._DATASET_COLUMNS:
            if column in working.columns:
                if working[column].isna().any():
                    raise ValueError(f"{column} cannot contain missing values")
                working[column] = working[column].map(
                    lambda value: str(value).strip()
                )
        ordered = self._ordered_with_segments(working)
        restored = ordered.sort_values("__feature_row_order__", kind="mergesort")
        return pd.Series(
            restored["__continuous_segment__"].to_numpy(dtype=np.int64),
            index=frame.index,
            name="continuous_segment",
        )

    @staticmethod
    def _optional_numeric_feature(
        frame: pd.DataFrame,
        feature_name: str,
    ) -> pd.Series:
        candidates, default = _OPTIONAL_FEATURES[feature_name]
        source_column = next(
            (column for column in candidates if column in frame.columns),
            None,
        )
        if source_column is None:
            return pd.Series(default, index=frame.index, dtype=float)
        return FeatureBuilder._numeric_column(
            frame,
            source_column,
            label=source_column,
        )

    def transform(self, frame: pd.DataFrame, physical_col: str) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if physical_col not in frame.columns:
            raise ValueError(f"missing physical weather column: {physical_col}")

        original_attrs = copy.deepcopy(frame.attrs)
        working = frame.copy(deep=True)
        working["tower_id"] = self._identity_values(working, "tower_id")
        if "line_id" in working.columns:
            working["line_id"] = self._identity_values(working, "line_id")
        working["timestamp"] = self._timestamps(working)
        working[physical_col] = self._numeric_column(
            working,
            physical_col,
            label=physical_col,
        )
        for column in self._DATASET_COLUMNS:
            if column in working.columns:
                if working[column].isna().any():
                    raise ValueError(f"{column} cannot contain missing values")
                working[column] = working[column].map(
                    lambda value: str(value).strip()
                )

        ordered = self._ordered_with_segments(working)
        previous = ordered.groupby(
            "__continuous_segment__", sort=False, dropna=False
        )[physical_col].shift(1)
        ordered["lag_1"] = previous.fillna(ordered[physical_col])

        timestamps = ordered["timestamp"]
        fractional_hour = (
            timestamps.dt.hour
            + timestamps.dt.minute / 60.0
            + timestamps.dt.second / 3600.0
        )
        hour_angle = 2.0 * np.pi * fractional_hour / 24.0
        day_angle = 2.0 * np.pi * (timestamps.dt.dayofyear - 1.0) / 365.25
        ordered["hour"] = timestamps.dt.hour
        ordered["day_of_year"] = timestamps.dt.dayofyear
        ordered["hour_sin"] = np.sin(hour_angle)
        ordered["hour_cos"] = np.cos(hour_angle)
        ordered["day_of_year_sin"] = np.sin(day_angle)
        ordered["day_of_year_cos"] = np.cos(day_angle)

        wind_direction = self._optional_numeric_feature(
            ordered, "wind_direction"
        ) % 360.0
        direction_angle = np.deg2rad(wind_direction)
        ordered["wind_direction_sin"] = np.sin(direction_angle)
        ordered["wind_direction_cos"] = np.cos(direction_angle)
        ordered["solar_radiation_feature"] = self._optional_numeric_feature(
            ordered, "solar_radiation"
        )
        ordered["humidity_feature"] = self._optional_numeric_feature(
            ordered, "humidity"
        )
        ordered["elevation_feature"] = self._optional_numeric_feature(
            ordered, "elevation"
        )
        ordered["slope_feature"] = self._optional_numeric_feature(
            ordered, "slope"
        )
        ordered["aspect_feature"] = (
            self._optional_numeric_feature(ordered, "aspect") % 360.0
        )

        result = ordered.sort_values(
            "__feature_row_order__", kind="mergesort"
        ).drop(
            columns=["__feature_row_order__", "__continuous_segment__"]
        )
        result.attrs = original_attrs
        return result


@dataclass
class ModelBundle:
    target_name: str
    feature_columns: list[str]
    model: object
    scaler: Optional[object] = None
    residual_bounds: Optional[tuple[float, float]] = None
    line_id: Optional[str] = None
    tower_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    cadence_minutes: float = float(DEFAULT_INTERVAL_MINUTES)


class ResidualPredictor:
    def __init__(self, bundles: Optional[dict] = None):
        self.bundles = bundles or {}
        self.feature_builder = FeatureBuilder()

    @classmethod
    def from_directory(cls, model_dir: Path):
        bundles = {}
        for bundle_path in model_dir.glob("*_bundle.joblib"):
            payload = joblib.load(bundle_path)
            bundles[payload["target_name"]] = ModelBundle(**payload)
        return cls(bundles)

    def build_features(
        self,
        df: pd.DataFrame,
        physical_col: str,
        cadence_minutes: Optional[Real] = None,
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        legacy_single_tower = "tower_id" not in df.columns
        feature_input = df.copy(deep=True)
        if legacy_single_tower:
            feature_input["tower_id"] = "__legacy_single_tower__"
        feature_builder = (
            self.feature_builder
            if cadence_minutes is None
            else FeatureBuilder(cadence_minutes=cadence_minutes)
        )
        features = feature_builder.transform(
            feature_input, physical_col=physical_col
        )
        if legacy_single_tower:
            features = features.drop(columns="tower_id")
        return features

    @staticmethod
    def _validated_physical(
        frame: pd.DataFrame, physical_col: str
    ) -> np.ndarray:
        if physical_col not in frame.columns:
            raise ValueError(f"missing physical weather column: {physical_col}")
        numeric = pd.to_numeric(frame[physical_col], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(numeric).all():
            raise ValueError(f"{physical_col} must contain finite values")
        return numeric

    @staticmethod
    def _fallback_output(
        output: pd.DataFrame,
        *,
        target_name: str,
        physical: np.ndarray,
        reason: str,
    ) -> pd.DataFrame:
        output[f"{target_name}_residual"] = np.zeros(len(output), dtype=float)
        output[f"{target_name}_final"] = physical.copy()
        output["used_ai"] = False
        output["fallback_reason"] = reason
        return output

    @staticmethod
    def _validate_bundle_scope(
        frame: pd.DataFrame, bundle: ModelBundle
    ) -> None:
        for column, expected in (
            ("line_id", bundle.line_id),
            ("tower_id", bundle.tower_id),
        ):
            if expected is None:
                continue
            if column not in frame.columns:
                raise ValueError(f"bundle requires {column}={expected}")
            values = frame[column]
            if values.isna().any() or set(values.map(str)) != {str(expected)}:
                raise ValueError(
                    f"bundle scope mismatch for {column}: expected {expected}"
                )

    def predict(
        self,
        df: pd.DataFrame,
        target_name: str,
        physical_col: str,
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if target_name not in _TARGET_PHYSICAL_COLUMNS:
            raise ValueError("target_name must be wind_speed or ambient_temp")
        if physical_col not in _TARGET_PHYSICAL_COLUMNS[target_name]:
            raise ValueError(
                "physical_col does not match the requested weather target"
            )
        output = df.copy(deep=True)
        physical = self._validated_physical(output, physical_col)
        bundle = self.bundles.get(target_name)
        if bundle is None:
            return self._fallback_output(
                output,
                target_name=target_name,
                physical=physical,
                reason="model_unavailable",
            )
        if bundle.target_name != target_name:
            raise ValueError("model target does not match requested target")
        self._validate_bundle_scope(output, bundle)

        features = self.build_features(
            output,
            physical_col,
            cadence_minutes=bundle.cadence_minutes,
        )
        missing = [
            column
            for column in bundle.feature_columns
            if column not in features.columns
        ]
        if missing:
            raise ValueError(f"missing model features: {', '.join(missing)}")
        feature_frame = features.loc[:, bundle.feature_columns]
        try:
            if bundle.scaler is not None:
                feature_frame = bundle.scaler.transform(feature_frame)
            raw_residual = np.asarray(
                bundle.model.predict(feature_frame), dtype=float
            ).reshape(-1)
            if len(raw_residual) != len(output):
                raise ValueError("model returned an unexpected prediction length")
        except Exception as exc:
            return self._fallback_output(
                output,
                target_name=target_name,
                physical=physical,
                reason=f"prediction_failed:{type(exc).__name__}",
            )

        residual = raw_residual.copy()
        if bundle.residual_bounds is not None:
            lower, upper = map(float, bundle.residual_bounds)
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError("residual_bounds must be finite and ordered")
            residual = np.clip(residual, lower, upper)

        finite_prediction = np.isfinite(raw_residual)
        residual[~finite_prediction] = 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            corrected = physical + residual
        finite_corrected = np.isfinite(corrected)
        valid_ai = finite_prediction & finite_corrected
        lower, upper = PHYSICAL_BOUNDS[target_name]
        bounds_exceeded = valid_ai & (
            (corrected < lower) | (corrected > upper)
        )
        valid_ai &= ~bounds_exceeded
        corrected[~valid_ai] = physical[~valid_ai]
        residual[~valid_ai] = 0.0

        reasons = np.full(len(output), "", dtype=object)
        reasons[~(finite_prediction & finite_corrected)] = (
            "non_finite_prediction"
        )
        reasons[bounds_exceeded] = "physical_bounds_exceeded"
        output[f"{target_name}_residual"] = residual
        output[f"{target_name}_final"] = corrected
        output["used_ai"] = valid_ai
        output["fallback_reason"] = reasons
        return output

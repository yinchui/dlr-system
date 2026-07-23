from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from config.config import CORRECTION_DEFAULTS
from modules.weather_upload import ensure_distinct_dataset_hashes


_REQUIRED_WEATHER_COLUMNS = (
    "tower_id",
    "timestamp",
    "ambient_temp",
    "wind_speed",
    "wind_direction",
    "dataset_role",
)

_INTERPOLATED_WEATHER_COLUMNS = {
    "ambient_temp",
    "wind_speed",
    "solar_radiation",
    "humidity",
    "elevation",
}

_GEOGRAPHY_OR_HEIGHT_MARKERS = (
    "longitude",
    "latitude",
    "elevation",
    "altitude",
    "height",
)

_SOURCE_HASH_LINEAGE_ATTR = "source_file_hashes"


@dataclass(frozen=True)
class AlignmentReport:
    physical_rows: int
    truth_rows: int
    matched_rows: int
    unmatched_rows: int
    coverage: float


def _require_dataframe(value, label: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} 必须是 pandas DataFrame")


def _require_columns(frame: pd.DataFrame, label: str) -> None:
    missing = [
        column
        for column in _REQUIRED_WEATHER_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{label} 缺少必需列: {', '.join(missing)}")


def _timestamp_timezone(frame: pd.DataFrame, label: str):
    timestamp_dtype = frame["timestamp"].dtype
    if not isinstance(timestamp_dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{label} timestamp 必须是带时区的时间列")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{label} timestamp 不能包含空值")
    return timestamp_dtype.tz


def _validate_weather_frame(frame: pd.DataFrame, label: str):
    _require_dataframe(frame, label)
    _require_columns(frame, label)
    timezone = _timestamp_timezone(frame, label)
    if frame["tower_id"].isna().any():
        raise ValueError(f"{label} tower_id 不能包含空值")
    return timezone


def _validate_resampling_role(source: pd.DataFrame) -> None:
    if source.empty:
        return
    if source["dataset_role"].isna().any() or not source[
        "dataset_role"
    ].eq("physical").all():
        raise ValueError(
            "重采样仅允许 dataset_role=physical；"
            "真实值必须通过 backward alignment 对齐"
        )


def _validate_alignment_role(
    frame: pd.DataFrame,
    *,
    label: str,
    expected_role: str,
) -> None:
    attr_role = frame.attrs.get("role")
    if attr_role not in (None, "") and attr_role != expected_role:
        raise ValueError(
            f"{label} attrs role 必须为 {expected_role}，实际为 {attr_role}"
        )
    if frame.empty:
        return
    roles = frame["dataset_role"]
    if roles.isna().any() or not roles.eq(expected_role).all():
        raise ValueError(
            f"{label} dataset_role 必须全部为 {expected_role}"
        )


def _lineage_attr_values(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _source_file_hash_lineage(frame: pd.DataFrame) -> tuple[str, ...]:
    values = list(
        _lineage_attr_values(frame.attrs.get(_SOURCE_HASH_LINEAGE_ATTR))
    )
    if "source_file_hash" in frame.columns:
        values.extend(frame["source_file_hash"].tolist())

    hashes = set()
    for value in values:
        if value is None:
            continue
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            continue
        text = str(value).strip()
        if text:
            hashes.add(text)
    return tuple(sorted(hashes))


def _dataset_hash_projection(dataset):
    if not isinstance(dataset, pd.DataFrame):
        return dataset
    return pd.DataFrame(
        {"source_file_hash": _source_file_hash_lineage(dataset)}
    )


def _interpolate_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    method = "time" if isinstance(numeric.index, pd.DatetimeIndex) else "linear"
    return numeric.interpolate(method=method, limit_area="inside")


def circular_interpolate(series: pd.Series) -> pd.Series:
    """Interpolate directions through unit vectors without extrapolating ends."""
    if not isinstance(series, pd.Series):
        raise TypeError("series 必须是 pandas Series")

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    finite = np.isfinite(numeric)
    normalized = numeric.where(finite) % 360.0
    radians = np.deg2rad(normalized)
    sine = pd.Series(np.sin(radians), index=series.index, dtype=float)
    cosine = pd.Series(np.cos(radians), index=series.index, dtype=float)

    sine = _interpolate_numeric(sine)
    cosine = _interpolate_numeric(cosine)
    magnitude = np.hypot(sine, cosine)
    defined = sine.notna() & cosine.notna() & (magnitude > 1e-12)

    result = pd.Series(np.nan, index=series.index, dtype=float, name=series.name)
    result.loc[defined] = (
        np.degrees(np.arctan2(sine.loc[defined], cosine.loc[defined])) % 360.0
    )
    result.loc[np.isclose(result, 360.0, equal_nan=False)] = 0.0
    return result


def _should_interpolate_numeric(column: str, series: pd.Series) -> bool:
    if column in _INTERPOLATED_WEATHER_COLUMNS:
        return True
    lower_name = column.lower()
    return pd.api.types.is_numeric_dtype(series.dtype) and any(
        marker in lower_name for marker in _GEOGRAPHY_OR_HEIGHT_MARKERS
    )


def _resample_one_tower(
    tower_frame: pd.DataFrame,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    ordered = tower_frame.sort_values("timestamp", kind="mergesort").copy()
    if len(ordered) == 1:
        return ordered.reset_index(drop=True)

    indexed = ordered.set_index("timestamp", drop=True)
    grid = pd.date_range(
        start=indexed.index[0],
        end=indexed.index[-1],
        freq=interval,
    )
    if grid[-1] != indexed.index[-1]:
        grid = grid.append(pd.DatetimeIndex([indexed.index[-1]]))
    interpolation_index = indexed.index.union(grid).sort_values()
    output = pd.DataFrame(index=grid)

    for column in ordered.columns:
        if column == "timestamp":
            continue
        values = indexed[column].reindex(interpolation_index)
        if column == "wind_direction":
            resampled = circular_interpolate(values)
        elif _should_interpolate_numeric(column, indexed[column]):
            resampled = _interpolate_numeric(values)
        else:
            resampled = values.ffill().bfill()
        output[column] = resampled.reindex(grid)

    timestamp_position = ordered.columns.get_loc("timestamp")
    output.insert(timestamp_position, "timestamp", grid)
    return output.loc[:, ordered.columns].reset_index(drop=True)


def resample_weather_by_tower(
    source: pd.DataFrame,
    interval_minutes: int = 30,
) -> pd.DataFrame:
    if (
        isinstance(interval_minutes, bool)
        or not isinstance(interval_minutes, Real)
        or not np.isfinite(interval_minutes)
        or interval_minutes <= 0
    ):
        raise ValueError("interval_minutes 必须是正数")

    _validate_weather_frame(source, "source")
    _validate_resampling_role(source)
    original_attrs = source.attrs.copy()
    source_hash_lineage = _source_file_hash_lineage(source)
    working = source.copy(deep=True)
    if working.empty:
        working.attrs = original_attrs
        working.attrs[_SOURCE_HASH_LINEAGE_ATTR] = source_hash_lineage
        return working

    duplicate_mask = working.duplicated(
        subset=["tower_id", "timestamp"], keep=False
    )
    if duplicate_mask.any():
        raise ValueError(
            "存在重复的 (tower_id, timestamp)，无法安全重采样"
        )

    interval = pd.Timedelta(minutes=float(interval_minutes))
    tower_results = [
        _resample_one_tower(tower_frame, interval)
        for _, tower_frame in working.groupby(
            "tower_id", sort=True, dropna=False
        )
    ]
    result = pd.concat(tower_results, ignore_index=True, sort=False)
    result = result.loc[:, source.columns]
    result = result.sort_values(
        ["tower_id", "timestamp"], kind="mergesort", ignore_index=True
    )
    result.attrs = original_attrs
    result.attrs[_SOURCE_HASH_LINEAGE_ATTR] = source_hash_lineage
    return result


def _validated_tolerance(tolerance) -> pd.Timedelta:
    try:
        value = pd.Timedelta(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tolerance 必须是非负 Timedelta") from exc
    if pd.isna(value) or value < pd.Timedelta(0):
        raise ValueError("tolerance 必须是非负 Timedelta")
    return value


def _validated_correction_parameter(value, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} 必须是有限数值")
    return numeric


def _unique_helper_column(columns, base: str) -> str:
    candidate = base
    while candidate in columns:
        candidate = f"_{candidate}"
    return candidate


def _merge_tower_frames(
    physical: pd.DataFrame,
    truth_without_tower: pd.DataFrame,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    left = physical.sort_values("timestamp", kind="mergesort")
    right = truth_without_tower.sort_values(
        "truth_timestamp", kind="mergesort"
    )
    return pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="truth_timestamp",
        direction="backward",
        tolerance=tolerance,
        suffixes=("_physical", "_truth"),
    )


def _height_column_name(
    *,
    side: str,
    physical_has_height: bool,
    truth_has_height: bool,
):
    if not (physical_has_height if side == "physical" else truth_has_height):
        return None
    if physical_has_height and truth_has_height:
        return f"measurement_height_{side}"
    return "measurement_height"


def _normalize_truth_measurement_height(
    aligned: pd.DataFrame,
    *,
    physical_has_height: bool,
    truth_has_height: bool,
    roughness_alpha: float,
    temp_lapse_rate: float,
) -> pd.DataFrame:
    result = aligned.copy()
    wind_truth = pd.to_numeric(result["wind_speed_truth"], errors="coerce")
    temp_truth = pd.to_numeric(result["ambient_temp_truth"], errors="coerce")
    result["wind_speed_truth_raw"] = wind_truth.copy()
    result["ambient_temp_truth_raw"] = temp_truth.copy()

    physical_height_column = _height_column_name(
        side="physical",
        physical_has_height=physical_has_height,
        truth_has_height=truth_has_height,
    )
    truth_height_column = _height_column_name(
        side="truth",
        physical_has_height=physical_has_height,
        truth_has_height=truth_has_height,
    )

    if truth_height_column is None:
        truth_height = pd.Series(np.nan, index=result.index, dtype=float)
    else:
        truth_height = pd.to_numeric(
            result[truth_height_column], errors="coerce"
        ).astype(float)
    result["measurement_height_truth_original"] = truth_height
    result["measurement_height_common"] = pd.Series(
        np.nan, index=result.index, dtype=float
    )
    result["height_normalized"] = False

    if physical_height_column is None or truth_height_column is None:
        available_height = (
            physical_height_column
            if physical_height_column is not None
            else truth_height_column
        )
        if available_height is not None:
            height = pd.to_numeric(
                result[available_height], errors="coerce"
            ).astype(float)
            assumed_same = (
                result["truth_timestamp"].notna()
                & np.isfinite(height)
                & (height > 0.0)
            )
            result.loc[assumed_same, "measurement_height_common"] = height.loc[
                assumed_same
            ]
        return result

    physical_height = pd.to_numeric(
        result[physical_height_column], errors="coerce"
    ).astype(float)
    valid_height = (
        result["truth_timestamp"].notna()
        & np.isfinite(physical_height)
        & np.isfinite(truth_height)
        & (physical_height > 0.0)
        & (truth_height > 0.0)
    )
    candidate_index = valid_height.loc[valid_height].index
    target_heights = physical_height.loc[candidate_index].to_numpy(dtype=float)
    truth_heights = truth_height.loc[candidate_index].to_numpy(dtype=float)
    raw_wind = wind_truth.loc[candidate_index].to_numpy(dtype=float)
    raw_temp = temp_truth.loc[candidate_index].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        height_ratio = np.divide(target_heights, truth_heights)
        candidate_wind = np.multiply(
            raw_wind,
            np.power(height_ratio, roughness_alpha),
        )
        candidate_temp = np.subtract(
            raw_temp,
            np.multiply(
                temp_lapse_rate,
                np.subtract(target_heights, truth_heights),
            ),
        )

    finite_candidates = np.isfinite(candidate_wind) & np.isfinite(candidate_temp)
    normalized_index = candidate_index[finite_candidates]
    result.loc[normalized_index, "wind_speed_truth"] = candidate_wind[
        finite_candidates
    ]
    result.loc[normalized_index, "ambient_temp_truth"] = candidate_temp[
        finite_candidates
    ]
    result.loc[normalized_index, "measurement_height_common"] = target_heights[
        finite_candidates
    ]
    result.loc[normalized_index, "height_normalized"] = True
    return result


def align_physical_and_truth(
    physical: pd.DataFrame,
    truth: pd.DataFrame,
    tolerance: pd.Timedelta,
    *,
    roughness_alpha: float = CORRECTION_DEFAULTS["roughness_alpha"],
    temp_lapse_rate: float = CORRECTION_DEFAULTS["temp_lapse_rate"],
) -> tuple[pd.DataFrame, AlignmentReport]:
    ensure_distinct_dataset_hashes(
        _dataset_hash_projection(physical),
        _dataset_hash_projection(truth),
    )
    physical_timezone = _validate_weather_frame(physical, "physical")
    _validate_weather_frame(truth, "truth")
    _validate_alignment_role(
        physical, label="physical", expected_role="physical"
    )
    _validate_alignment_role(truth, label="truth", expected_role="truth")
    tolerance = _validated_tolerance(tolerance)
    roughness_alpha = _validated_correction_parameter(
        roughness_alpha, "roughness_alpha"
    )
    temp_lapse_rate = _validated_correction_parameter(
        temp_lapse_rate, "temp_lapse_rate"
    )

    physical_rows = len(physical)
    truth_rows = len(truth)
    physical_work = physical.copy(deep=True)
    truth_work = truth.copy(deep=True)
    truth_work["timestamp"] = truth_work["timestamp"].dt.tz_convert(
        physical_timezone
    )

    physical_has_height = "measurement_height" in physical_work.columns
    truth_has_height = "measurement_height" in truth_work.columns
    order_column = _unique_helper_column(
        set(physical_work.columns) | set(truth_work.columns),
        "__physical_order__",
    )
    physical_work[order_column] = np.arange(physical_rows, dtype=np.int64)
    truth_prepared = truth_work.rename(
        columns={"timestamp": "truth_timestamp"}
    )

    merged_towers = []
    for tower_id, physical_tower in physical_work.groupby(
        "tower_id", sort=False, dropna=False
    ):
        truth_tower = truth_prepared.loc[
            truth_prepared["tower_id"] == tower_id
        ].drop(columns="tower_id")
        merged_towers.append(
            _merge_tower_frames(physical_tower, truth_tower, tolerance)
        )

    if merged_towers:
        aligned = pd.concat(merged_towers, ignore_index=True, sort=False)
    else:
        empty_truth = truth_prepared.iloc[0:0].drop(columns="tower_id")
        aligned = _merge_tower_frames(
            physical_work.iloc[0:0], empty_truth, tolerance
        )

    aligned = aligned.sort_values(
        order_column, kind="mergesort", ignore_index=True
    ).drop(columns=order_column)
    aligned = _normalize_truth_measurement_height(
        aligned,
        physical_has_height=physical_has_height,
        truth_has_height=truth_has_height,
        roughness_alpha=roughness_alpha,
        temp_lapse_rate=temp_lapse_rate,
    )

    matched_rows = int(aligned["truth_timestamp"].notna().sum())
    unmatched_rows = physical_rows - matched_rows
    coverage = matched_rows / physical_rows if physical_rows else 0.0
    report = AlignmentReport(
        physical_rows=physical_rows,
        truth_rows=truth_rows,
        matched_rows=matched_rows,
        unmatched_rows=unmatched_rows,
        coverage=coverage,
    )
    return aligned, report

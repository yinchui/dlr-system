from collections import Counter
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd

from config.config import PROJECT_TIMEZONE
from modules.data_processor import (
    CANONICAL_WEATHER_COLUMNS,
    CanonicalWeatherResult,
    DataQualityReport,
    canonicalize_weather_frame,
)


@dataclass(frozen=True)
class UploadBlob:
    name: str
    content: bytes
    sha256: str


@dataclass(frozen=True)
class NormalizedWeatherFile:
    name: str
    sha256: str
    frame: pd.DataFrame
    report: DataQualityReport


@dataclass(frozen=True)
class WeatherUploadResult:
    frame: pd.DataFrame
    report: DataQualityReport
    files: tuple[NormalizedWeatherFile, ...]


@dataclass(frozen=True)
class OptionalTruthNormalization:
    snapshot: Optional[WeatherUploadResult]
    warning: Optional[str]


def freeze_uploaded_file(uploaded_file) -> UploadBlob:
    raw_content = uploaded_file.getvalue()
    try:
        content = bytes(raw_content)
    except (TypeError, ValueError) as exc:
        raise TypeError("上传文件内容必须可转换为 bytes") from exc
    return UploadBlob(
        name=uploaded_file.name,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def freeze_uploaded_files(uploaded_files) -> tuple[UploadBlob, ...]:
    return tuple(freeze_uploaded_file(uploaded_file) for uploaded_file in uploaded_files)


def _read_csv(content: bytes) -> pd.DataFrame:
    decode_errors = []
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError as exc:
            decode_errors.append(exc)
    raise ValueError(
        "CSV 编码无法识别，支持 utf-8-sig、gb18030 和 gbk"
    ) from decode_errors[-1]


def _read_weather_blob(blob: UploadBlob) -> pd.DataFrame:
    extension = Path(blob.name).suffix.lower()
    if extension == ".csv":
        return _read_csv(blob.content)
    if extension in {".xlsx", ".xlsm"}:
        return pd.read_excel(BytesIO(blob.content))
    display_extension = extension or "无扩展名"
    raise ValueError(f"不支持的气象文件格式: {display_extension}")


def _empty_canonical_frame(timezone: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=CANONICAL_WEATHER_COLUMNS)
    frame["timestamp"] = pd.Series(
        [], dtype=pd.DatetimeTZDtype(tz=timezone)
    )
    return frame


def _merge_reports(
    results: list[CanonicalWeatherResult],
    cross_file_duplicates: int,
    valid_rows: int,
) -> DataQualityReport:
    input_rows = sum(result.report.input_rows for result in results)
    reasons = Counter()
    duplicate_rows = cross_file_duplicates
    for result in results:
        reasons.update(result.report.reasons)
        duplicate_rows += result.report.duplicate_rows
    if cross_file_duplicates:
        reasons["duplicate_tower_timestamp"] += cross_file_duplicates
    return DataQualityReport(
        input_rows=input_rows,
        valid_rows=valid_rows,
        dropped_rows=input_rows - valid_rows,
        duplicate_rows=duplicate_rows,
        reasons=dict(reasons),
    )


def normalize_uploaded_weather_files(
    uploaded_files,
    role: str,
    timezone: str = PROJECT_TIMEZONE,
) -> WeatherUploadResult:
    if role not in {"physical", "truth"}:
        raise ValueError("role 仅允许 physical 或 truth")

    blobs = freeze_uploaded_files(uploaded_files)
    canonical_results: list[CanonicalWeatherResult] = []
    normalized_files = []
    for blob in blobs:
        raw_frame = _read_weather_blob(blob)
        canonical = canonicalize_weather_frame(
            raw_frame,
            role=role,
            timezone=timezone,
            source_hash=blob.sha256,
        )
        canonical_results.append(canonical)
        normalized_files.append(
            NormalizedWeatherFile(
                name=blob.name,
                sha256=blob.sha256,
                frame=canonical.frame,
                report=canonical.report,
            )
        )

    if canonical_results:
        frame = pd.concat(
            [result.frame for result in canonical_results],
            ignore_index=True,
            sort=False,
        )
        duplicate_mask = frame.duplicated(
            subset=["tower_id", "timestamp"], keep="first"
        )
        cross_file_duplicates = int(duplicate_mask.sum())
        if cross_file_duplicates:
            frame = frame.loc[~duplicate_mask].reset_index(drop=True)
    else:
        frame = _empty_canonical_frame(timezone)
        cross_file_duplicates = 0

    report = _merge_reports(
        canonical_results,
        cross_file_duplicates=cross_file_duplicates,
        valid_rows=len(frame),
    )
    return WeatherUploadResult(
        frame=frame,
        report=report,
        files=tuple(normalized_files),
    )


def normalize_optional_truth_weather(
    uploaded_files,
    *,
    ai_enabled: bool,
    normalizer=None,
) -> OptionalTruthNormalization:
    if not ai_enabled or not uploaded_files:
        return OptionalTruthNormalization(snapshot=None, warning=None)

    normalize = normalizer or normalize_uploaded_weather_files
    try:
        snapshot = normalize(uploaded_files, role="truth")
    except Exception as exc:
        return OptionalTruthNormalization(
            snapshot=None,
            warning=(
                "真实气象数据解析失败，已跳过 AI 训练并继续物理 DLR "
                f"计算：{exc}"
            ),
        )
    return OptionalTruthNormalization(snapshot=snapshot, warning=None)


def _extract_source_hashes(dataset) -> set[str]:
    if dataset is None:
        return set()
    if isinstance(dataset, WeatherUploadResult):
        return {item.sha256 for item in dataset.files if item.sha256}
    if isinstance(dataset, NormalizedWeatherFile):
        return {dataset.sha256} if dataset.sha256 else set()
    if isinstance(dataset, pd.DataFrame):
        if "source_file_hash" not in dataset.columns:
            return set()
        values = dataset["source_file_hash"]
    elif isinstance(dataset, str):
        values = [dataset]
    else:
        values = dataset

    hashes = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            hashes.add(text)
    return hashes


def ensure_distinct_dataset_hashes(physical, truth) -> None:
    shared_hashes = _extract_source_hashes(physical) & _extract_source_hashes(truth)
    if shared_hashes:
        shared = ", ".join(sorted(shared_hashes))
        raise ValueError(
            f"同一文件哈希不能同时作为 physical 与 truth 数据: {shared}"
        )

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np
from filelock import FileLock


_DEFAULT_LOG_NAME = "dlr-audit.jsonl"


def _require_trace_text(value: str, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_file_component(value: str, name: str) -> None:
    _require_trace_text(value, name)
    if (
        value != value.strip()
        or value in {".", ".."}
        or Path(value).is_absolute()
        or PureWindowsPath(value).drive
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be a safe file name component")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        default=_json_default,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True)
class AuditEvent:
    run_id: str
    result_id: str
    line_id: str
    tower_id: str
    stage: str
    input_hash: str
    config_hash: str
    source: str
    fallback_reason: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error_code: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "result_id",
            "line_id",
            "tower_id",
            "stage",
            "input_hash",
            "config_hash",
            "source",
        ):
            _require_trace_text(getattr(self, name), name)
        _require_trace_text(
            self.fallback_reason,
            "fallback_reason",
            allow_empty=True,
        )
        _require_trace_text(self.error_code, "error_code", allow_empty=True)
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "result_id": self.result_id,
            "line_id": self.line_id,
            "tower_id": self.tower_id,
            "stage": self.stage,
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "source": self.source,
            "fallback_reason": self.fallback_reason,
            "timestamp": self.timestamp,
            "error_code": self.error_code,
            "details": dict(self.details),
        }

    @classmethod
    def example(cls) -> "AuditEvent":
        return cls(
            run_id="example-run",
            result_id="example-result",
            line_id="example-line",
            tower_id="example-tower",
            stage="example",
            input_hash="example-input-hash",
            config_hash="example-config-hash",
            source="example",
            fallback_reason="",
        )


class JsonAuditLogger:
    def __init__(
        self,
        output_dir: Path | str,
        *,
        log_name: str = _DEFAULT_LOG_NAME,
    ) -> None:
        _validate_file_component(log_name, "log_name")
        if not log_name.endswith(".jsonl"):
            raise ValueError("log_name must end with .jsonl")
        self.output_dir = Path(output_dir).expanduser()
        self.log_path = self.output_dir / log_name
        self.lock_path = self.output_dir / f".{log_name}.lock"

    def write(self, event: AuditEvent) -> bool:
        try:
            if not isinstance(event, AuditEvent):
                raise TypeError("event must be an AuditEvent")
            line = _json_bytes(event.to_dict()) + b"\n"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(self.lock_path):
                with self.log_path.open("a+b") as stream:
                    stream.seek(0, os.SEEK_END)
                    original_size = stream.tell()
                    try:
                        written = stream.write(line)
                        if written != len(line):
                            raise OSError("incomplete audit write")
                        stream.flush()
                        os.fsync(stream.fileno())
                    except Exception:
                        stream.truncate(original_size)
                        stream.flush()
                        os.fsync(stream.fileno())
                        raise
            return True
        except Exception:
            return False


def write_result_atomic(
    output_dir: Path | str,
    result_id: str,
    payload: Any,
) -> Path:
    _validate_file_component(result_id, "result_id")
    serialized = _json_bytes(payload)
    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{result_id}.json"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{result_id}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        temp_path = None
        _fsync_directory(target.parent)
        return target
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

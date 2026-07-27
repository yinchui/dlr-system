from __future__ import annotations

import hashlib
import io
import re
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import joblib

from modules.ai_prediction import ModelBundle
from modules.model_registry import (
    ModelAttempt,
    ModelCandidate,
    ModelKey,
    ModelMetadata,
    ModelRegistry,
    PromotionDecision,
    UnsafeModelPathError,
    _DETERMINISTIC_REJECTION_REASONS,
)


_MODEL_BUCKET_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_MODEL_STATUSES = frozenset({"active_provisional", "active"})
_SUPABASE_SERVICE_TIMEOUT_SECONDS = 5.0
_GENERATION_COLUMNS = (
    "generation_id,project_id,line_id,tower_id,target,model_version,"
    "storage_path,model_checksum,metadata,status"
)
_REJECTION_CONFLICT_COLUMNS = (
    "project_id,line_id,tower_id,target,attempt_fingerprint"
)


class SupabaseTransportError(OSError):
    """Sanitized SDK request or storage transfer failure."""


def _normalized_status_code(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return None


def _exception_status_code(exc: Exception) -> Optional[int]:
    for name in ("status", "status_code", "statusCode"):
        status = _normalized_status_code(getattr(exc, name, None))
        if status is not None:
            return status
    response = getattr(exc, "response", None)
    status = _normalized_status_code(
        getattr(response, "status_code", None)
    )
    if status is not None:
        return status
    for value in exc.args:
        if not isinstance(value, Mapping):
            continue
        for name in ("status", "status_code", "statusCode"):
            status = _normalized_status_code(value.get(name))
            if status is not None:
                return status
    return None


def _scope(key: ModelKey) -> dict[str, str]:
    if not isinstance(key, ModelKey):
        raise TypeError("key must be a ModelKey")
    return {
        "project_id": key.project_id,
        "line_id": key.line_id,
        "tower_id": key.tower_id,
        "target": key.target,
    }


def _validated_uuid(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise OSError(f"invalid Supabase model {name} uuid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise OSError(f"invalid Supabase model {name} uuid") from None
    if str(parsed) != value.lower():
        raise OSError(f"invalid Supabase model {name} uuid")
    return str(parsed)


def _validated_rows(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(
        not isinstance(row, Mapping) for row in value
    ):
        raise OSError(f"invalid Supabase model {name} response")
    return value


def _execute(request_factory: Callable[[], Any], operation: str) -> Any:
    try:
        response = request_factory().execute()
    except Exception:
        raise SupabaseTransportError(
            f"Supabase model {operation} failed"
        ) from None
    try:
        return response.data
    except Exception:
        raise OSError(
            f"invalid Supabase model {operation} response"
        ) from None


def _artifact_path(generation_id: str, key: ModelKey) -> str:
    generation_id = _validated_uuid(generation_id, "generation")
    values = _scope(key)
    return (
        f"{values['project_id']}/{values['line_id']}/"
        f"{values['tower_id']}/{values['target']}/"
        f"{generation_id}/model.joblib"
    )


@dataclass(frozen=True)
class RemoteGeneration:
    generation_id: str
    key: ModelKey
    model_version: str
    storage_path: str
    model_checksum: str
    metadata: ModelMetadata
    status: str
    revision: int

    def __post_init__(self) -> None:
        generation_id = _validated_uuid(self.generation_id, "generation")
        object.__setattr__(self, "generation_id", generation_id)
        if not isinstance(self.key, ModelKey):
            raise TypeError("key must be a ModelKey")
        if not isinstance(self.metadata, ModelMetadata):
            raise TypeError("metadata must be ModelMetadata")
        if self.metadata.key != self.key:
            raise ValueError("remote metadata scope does not match key")
        if self.model_version != self.metadata.model_version:
            raise ValueError("remote model version does not match metadata")
        if self.status not in _REMOTE_MODEL_STATUSES:
            raise ValueError("unsupported remote model status")
        if self.status != self.metadata.status:
            raise ValueError("remote model status does not match metadata")
        if (
            not isinstance(self.model_checksum, str)
            or _SHA256_PATTERN.fullmatch(self.model_checksum) is None
            or self.model_checksum != self.metadata.checksum
        ):
            raise ValueError("remote model checksum does not match metadata")
        if self.storage_path != _artifact_path(generation_id, self.key):
            raise ValueError("remote model storage path does not match scope")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise ValueError("remote model revision must be a positive integer")


class SupabaseModelStore:
    """Validated Supabase I/O for immutable model generations."""

    def __init__(self, client: Any, *, bucket: str = "dlr-models") -> None:
        if client is None:
            raise TypeError("client is required")
        if (
            not isinstance(bucket, str)
            or _MODEL_BUCKET_PATTERN.fullmatch(bucket) is None
        ):
            raise ValueError("bucket must be a safe Supabase bucket name")
        self._client = client
        self.bucket = bucket

    @classmethod
    def from_credentials(
        cls,
        url: str,
        secret_key: str,
        *,
        bucket: str = "dlr-models",
    ) -> "SupabaseModelStore":
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Supabase URL is required")
        if not isinstance(secret_key, str) or not secret_key.strip():
            raise ValueError("Supabase secret key is required")
        try:
            from supabase import ClientOptions, create_client

            options = ClientOptions(
                postgrest_client_timeout=_SUPABASE_SERVICE_TIMEOUT_SECONDS,
                storage_client_timeout=_SUPABASE_SERVICE_TIMEOUT_SECONDS,
                function_client_timeout=_SUPABASE_SERVICE_TIMEOUT_SECONDS,
            )
            client = create_client(
                url.strip(),
                secret_key.strip(),
                options=options,
            )
        except Exception:
            raise OSError("Supabase model client initialization failed") from None
        return cls(client, bucket=bucket)

    def current(self, key: ModelKey) -> Optional[RemoteGeneration]:
        scope = _scope(key)
        head_data = _execute(
            lambda: self._client.table("dlr_model_heads")
            .select(
                "generation_id,revision,project_id,line_id,tower_id,target"
            )
            .match(scope),
            "head query",
        )
        head_rows = _validated_rows(head_data, "head")
        if not head_rows:
            return None
        if len(head_rows) != 1:
            raise OSError("invalid Supabase model head response")
        head = head_rows[0]
        generation_id = _validated_uuid(
            head.get("generation_id"), "head generation"
        )
        revision = head.get("revision")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            raise OSError("invalid Supabase model head revision")
        if any(head.get(name) != value for name, value in scope.items()):
            raise OSError("invalid Supabase model head scope")

        generation_data = _execute(
            lambda: self._client.table("dlr_model_generations")
            .select(_GENERATION_COLUMNS)
            .eq("generation_id", generation_id),
            "generation query",
        )
        generation_rows = _validated_rows(generation_data, "generation")
        if len(generation_rows) != 1:
            raise OSError("invalid Supabase model generation response")
        row = generation_rows[0]
        try:
            metadata = ModelMetadata.from_dict(row["metadata"])
        except Exception:
            raise OSError("invalid Supabase model metadata") from None
        try:
            row_key = ModelKey(
                project_id=row["project_id"],
                line_id=row["line_id"],
                tower_id=row["tower_id"],
                target=row["target"],
            )
            generation = RemoteGeneration(
                generation_id=row["generation_id"],
                key=row_key,
                model_version=row["model_version"],
                storage_path=row["storage_path"],
                model_checksum=row["model_checksum"],
                metadata=metadata,
                status=row["status"],
                revision=revision,
            )
        except (KeyError, TypeError, ValueError, OSError):
            raise OSError("invalid Supabase model generation scope or path") from None
        if generation.generation_id != generation_id or generation.key != key:
            raise OSError("invalid Supabase model generation scope or path")
        return generation

    def download(self, generation: RemoteGeneration) -> bytes:
        if not isinstance(generation, RemoteGeneration):
            raise TypeError("generation must be a RemoteGeneration")
        try:
            artifact = (
                self._client.storage.from_(self.bucket)
                .download(generation.storage_path)
            )
        except Exception as exc:
            if _exception_status_code(exc) == 404:
                raise OSError(
                    "Supabase model download object is missing"
                ) from None
            raise SupabaseTransportError(
                "Supabase model download failed"
            ) from None
        if not isinstance(artifact, bytes):
            raise OSError("Supabase model download returned invalid bytes")
        if hashlib.sha256(artifact).hexdigest() != generation.model_checksum:
            raise OSError("Supabase model download checksum mismatch")
        return artifact

    def upload(
        self,
        generation_id: str,
        key: ModelKey,
        artifact: bytes,
    ) -> str:
        path = _artifact_path(generation_id, key)
        if not isinstance(artifact, bytes) or not artifact:
            raise ValueError("artifact must be non-empty bytes")
        try:
            response = self._client.storage.from_(self.bucket).upload(
                path,
                artifact,
                file_options={
                    "content-type": "application/octet-stream",
                    "upsert": "false",
                },
            )
        except Exception:
            raise SupabaseTransportError(
                "Supabase model upload failed"
            ) from None
        try:
            response_path = (
                response.get("path")
                if isinstance(response, Mapping)
                else getattr(response, "path")
            )
        except Exception:
            raise OSError("invalid Supabase model upload response") from None
        if response_path != path:
            raise OSError("invalid Supabase model upload response")
        return path

    def activate(
        self,
        generation_id: str,
        key: ModelKey,
        metadata: ModelMetadata,
        storage_path: str,
        *,
        expected_generation_id: Optional[str],
    ) -> bool:
        generation_id = _validated_uuid(generation_id, "generation")
        scope = _scope(key)
        if not isinstance(metadata, ModelMetadata) or metadata.key != key:
            raise ValueError("metadata scope does not match key")
        expected_path = _artifact_path(generation_id, key)
        if storage_path != expected_path:
            raise ValueError("storage path does not match generation scope")
        if expected_generation_id is not None:
            expected_generation_id = _validated_uuid(
                expected_generation_id, "expected generation"
            )
        payload = {
            "p_generation_id": generation_id,
            "p_project_id": scope["project_id"],
            "p_line_id": scope["line_id"],
            "p_tower_id": scope["tower_id"],
            "p_target": scope["target"],
            "p_model_version": metadata.model_version,
            "p_storage_path": storage_path,
            "p_model_checksum": metadata.checksum,
            "p_metadata": metadata.to_dict(),
            "p_status": metadata.status,
            "p_expected_generation_id": expected_generation_id,
        }
        result = _execute(
            lambda: self._client.rpc(
                "activate_dlr_model_generation", payload
            ),
            "activation",
        )
        if type(result) is not bool:
            raise OSError("invalid Supabase model activation response")
        return result

    def was_rejected(self, attempt: ModelAttempt) -> bool:
        if not isinstance(attempt, ModelAttempt):
            raise TypeError("attempt must be a ModelAttempt")
        lookup = {
            **_scope(attempt.key),
            "attempt_fingerprint": attempt.fingerprint,
        }
        data = _execute(
            lambda: self._client.table("dlr_model_rejections")
            .select(
                "project_id,line_id,tower_id,target,attempt_fingerprint"
            )
            .match(lookup),
            "rejection query",
        )
        rows = _validated_rows(data, "rejection")
        if not rows:
            return False
        if len(rows) != 1 or any(
            rows[0].get(name) != value for name, value in lookup.items()
        ):
            raise OSError("invalid Supabase model rejection response")
        return True

    def record_rejection(self, attempt: ModelAttempt, reason: str) -> None:
        if not isinstance(attempt, ModelAttempt):
            raise TypeError("attempt must be a ModelAttempt")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        row = {
            **_scope(attempt.key),
            "attempt_fingerprint": attempt.fingerprint,
            "champion_context_hash": attempt.champion_context_hash,
            "reason": reason,
            "attempt": attempt.to_dict(),
        }
        data = _execute(
            lambda: self._client.table("dlr_model_rejections").upsert(
                row,
                on_conflict=_REJECTION_CONFLICT_COLUMNS,
            ),
            "rejection write",
        )
        rows = _validated_rows(data, "rejection write")
        expected = {
            **_scope(attempt.key),
            "attempt_fingerprint": attempt.fingerprint,
        }
        if len(rows) != 1 or any(
            rows[0].get(name) != value for name, value in expected.items()
        ):
            raise OSError("invalid Supabase model rejection write response")


class SupabaseModelRegistry(ModelRegistry):
    """Remote-authoritative model registry with a private local cache."""

    def __init__(
        self,
        store: SupabaseModelStore,
        *,
        cache_dir: Path | str | None = None,
        min_mae_improvement: float = 0.0,
        max_generations: int = 2,
        max_attempt_records: int = 64,
        audit_logger: Optional[Any] = None,
        audit_run_id: Optional[str] = None,
        audit_result_id: Optional[str] = None,
        audit_source: str = "supabase_model_registry",
    ) -> None:
        if store is None:
            raise TypeError("store is required")
        self.store = store
        self._temporary_cache: Optional[tempfile.TemporaryDirectory[str]] = None
        if cache_dir is None:
            self._temporary_cache = tempfile.TemporaryDirectory(
                prefix="dlr-model-registry-"
            )
            cache_dir = self._temporary_cache.name
        self._remote_generation_ids: dict[ModelKey, Optional[str]] = {}
        self._hydrated_generation_ids: dict[ModelKey, str] = {}
        self._hydrated_local_generations: dict[ModelKey, Path] = {}
        self._pipeline_run_active = False
        self._pipeline_transport_failed = False
        super().__init__(
            cache_dir,
            min_mae_improvement=min_mae_improvement,
            max_generations=max_generations,
            max_attempt_records=max_attempt_records,
            audit_logger=audit_logger,
            audit_run_id=audit_run_id,
            audit_result_id=audit_result_id,
            audit_source=audit_source,
        )

    def _load_many_should_abort(self, exc: Exception) -> bool:
        return isinstance(exc, SupabaseTransportError)

    def begin_pipeline_run(self) -> None:
        self._pipeline_run_active = True
        self._pipeline_transport_failed = False

    def end_pipeline_run(self) -> None:
        self._pipeline_run_active = False
        self._pipeline_transport_failed = False

    def model_operations_available(self) -> bool:
        return not (
            self._pipeline_run_active and self._pipeline_transport_failed
        )

    def _trip_pipeline_transport_circuit(self) -> None:
        if self._pipeline_run_active:
            self._pipeline_transport_failed = True

    def _remote_call(self, operation: Callable[[], Any]) -> Any:
        if not self.model_operations_available():
            raise SupabaseTransportError(
                "Supabase model transport circuit is open"
            )
        try:
            return operation()
        except SupabaseTransportError:
            self._trip_pipeline_transport_circuit()
            raise

    def _cached_generation_is_current(
        self,
        key: ModelKey,
        remote: RemoteGeneration,
    ) -> bool:
        if self._hydrated_generation_ids.get(key) != remote.generation_id:
            return False
        cached_path = self._hydrated_local_generations.get(key)
        if cached_path is None:
            return False
        target_dir = self._target_dir(key)
        if not target_dir.is_symlink():
            if target_dir.exists() or target_dir.is_symlink():
                raise UnsafeModelPathError("active model pointer is unsafe")
            return False
        active_path = self._active_generation(key)
        if active_path != cached_path:
            return False
        header, _ = self._read_generation_header(key, active_path)
        return bool(
            header is not None
            and header.metadata == remote.metadata
            and header.model_checksum == remote.model_checksum
            and self._generation_model_checksum_is_valid(header)
        )

    def _hydrate_current_locked(
        self, key: ModelKey
    ) -> Optional[RemoteGeneration]:
        self._remote_generation_ids.pop(key, None)
        remote = self._remote_call(lambda: self.store.current(key))
        if remote is not None and not isinstance(remote, RemoteGeneration):
            raise OSError("invalid Supabase model current generation")
        self._remote_generation_ids[key] = (
            None if remote is None else remote.generation_id
        )
        if remote is None:
            self._hydrated_generation_ids.pop(key, None)
            self._hydrated_local_generations.pop(key, None)
            return None
        if self._cached_generation_is_current(key, remote):
            return remote

        artifact = self._remote_call(lambda: self.store.download(remote))
        try:
            bundle = joblib.load(io.BytesIO(artifact))
        except Exception:
            raise OSError("Supabase model artifact deserialization failed") from None
        if not isinstance(bundle, ModelBundle):
            raise OSError("Supabase model artifact has an invalid bundle")
        try:
            candidate = ModelCandidate(
                key=key,
                bundle=bundle,
                metadata=remote.metadata,
            )
            self._validate_promotion_contract(candidate, None)
        except (TypeError, ValueError, OverflowError):
            raise OSError("Supabase model artifact validation failed") from None
        try:
            active_metadata, _ = super()._publish_locked(
                candidate,
                status=remote.status,
                serialized_model=artifact,
            )
        except UnsafeModelPathError:
            raise
        except Exception:
            raise OSError("Supabase model cache publication failed") from None
        active_path = self._active_generation(key)
        header, reason = self._read_generation_header(key, active_path)
        if (
            header is None
            or reason
            or active_metadata != remote.metadata
            or header.metadata != remote.metadata
            or header.model_checksum != remote.model_checksum
            or not self._generation_model_checksum_is_valid(header)
        ):
            raise OSError("Supabase model cache checksum mismatch")
        self._hydrated_generation_ids[key] = remote.generation_id
        self._hydrated_local_generations[key] = active_path
        return remote

    def _load_locked(
        self,
        key,
        expected_compatibility,
        expected_training_contract_hash,
        expected_backend_id,
    ):
        if self._hydrate_current_locked(key) is None:
            return self._fallback("model_not_found")
        return super()._load_locked(
            key,
            expected_compatibility,
            expected_training_contract_hash,
            expected_backend_id,
        )

    def _load_current_locked(self, key):
        try:
            return super()._load_current_locked(key)
        except (SupabaseTransportError, UnsafeModelPathError):
            raise
        except OSError:
            if key not in self._remote_generation_ids:
                raise
            # The head is trusted enough for CAS, but its artifact is unusable.
            return self._fallback("corrupt_model")

    def _current_generation_header_locked(self, key):
        if self._hydrate_current_locked(key) is None:
            return None, "model_not_found"
        return super()._current_generation_header_locked(key)

    def was_rejected(self, attempt: ModelAttempt) -> bool:
        if not isinstance(attempt, ModelAttempt):
            raise TypeError("attempt must be a ModelAttempt")
        try:
            with self._lock_for(attempt.key):
                current = self._load_current_metadata_locked(attempt.key)
                champion = (
                    current.metadata if current.metadata is not None else None
                )
                current_attempt = replace(
                    attempt,
                    champion_context_hash=self._champion_context_hash(champion),
                )
                return self._remote_call(
                    lambda: self.store.was_rejected(current_attempt)
                )
        except Exception:
            return False

    def _record_attempt_rejection_locked(
        self,
        attempt: ModelAttempt,
        reason: str,
    ) -> None:
        if reason not in _DETERMINISTIC_REJECTION_REASONS:
            raise ValueError("only deterministic rejections may be persisted")
        self._remote_call(lambda: self.store.record_rejection(attempt, reason))

    @staticmethod
    def _remote_failure(
        reason: str,
        champion: Optional[ModelMetadata],
        local: PromotionDecision,
    ) -> PromotionDecision:
        return PromotionDecision(
            False,
            reason,
            champion,
            audit_persisted=local.audit_persisted,
        )

    def _remember_activation(self, key: ModelKey, generation_id: str) -> None:
        active_path = self._active_generation(key)
        self._remote_generation_ids[key] = generation_id
        self._hydrated_generation_ids[key] = generation_id
        self._hydrated_local_generations[key] = active_path

    def _serialize_remote_candidate(
        self,
        candidate: ModelCandidate,
        *,
        status: str,
    ) -> tuple[ModelCandidate, bytes]:
        buffer = io.BytesIO()
        joblib.dump(candidate.bundle, buffer, compress=3)
        artifact = buffer.getvalue()
        if not artifact:
            raise ValueError("serialized candidate is empty")
        metadata = replace(
            candidate.metadata,
            checksum=hashlib.sha256(artifact).hexdigest(),
            status=status,
        )
        bundle = joblib.load(io.BytesIO(artifact))
        staged = ModelCandidate(
            key=candidate.key,
            bundle=bundle,
            metadata=metadata,
        )
        self._validate_promotion_contract(staged, None)
        return staged, artifact

    def _mark_remote_activation_for_hydration(
        self,
        key: ModelKey,
        generation_id: str,
    ) -> None:
        self._remote_generation_ids[key] = generation_id
        self._hydrated_generation_ids.pop(key, None)
        self._hydrated_local_generations.pop(key, None)

    def _publish_decision(
        self,
        candidate: ModelCandidate,
        *,
        status: str,
        reason: str,
        champion: Optional[ModelMetadata] = None,
    ) -> PromotionDecision:
        if not self.model_operations_available():
            return PromotionDecision(
                False,
                "remote_transport_unavailable",
                champion,
            )
        if candidate.key not in self._remote_generation_ids:
            return PromotionDecision(False, "remote_state_unavailable", champion)
        expected_generation_id = self._remote_generation_ids[candidate.key]
        try:
            staged, artifact = self._serialize_remote_candidate(
                candidate,
                status=status,
            )
        except Exception:
            return PromotionDecision(
                False,
                "remote_candidate_validation_failed",
                champion,
            )
        remote_decision = PromotionDecision(True, reason, staged.metadata)

        generation_id = str(uuid.uuid4())
        try:
            storage_path = self._remote_call(
                lambda: self.store.upload(
                    generation_id,
                    candidate.key,
                    artifact,
                )
            )
        except OSError:
            return self._remote_failure(
                "remote_upload_failed", champion, remote_decision
            )
        activated = False
        try:
            activated = self.store.activate(
                generation_id,
                candidate.key,
                staged.metadata,
                storage_path,
                expected_generation_id=expected_generation_id,
            )
        except SupabaseTransportError:
            try:
                reconciled = self.store.current(candidate.key)
            except SupabaseTransportError:
                self._trip_pipeline_transport_circuit()
                reconciled = None
            except OSError:
                reconciled = None
            if (
                isinstance(reconciled, RemoteGeneration)
                and reconciled.generation_id == generation_id
                and reconciled.key == candidate.key
                and reconciled.storage_path == storage_path
                and reconciled.model_checksum == staged.metadata.checksum
                and reconciled.metadata == staged.metadata
            ):
                activated = True
            else:
                self._trip_pipeline_transport_circuit()
                return self._remote_failure(
                    "remote_activation_failed", champion, remote_decision
                )
        except OSError:
            try:
                reconciled = self._remote_call(
                    lambda: self.store.current(candidate.key)
                )
            except OSError:
                reconciled = None
            if (
                isinstance(reconciled, RemoteGeneration)
                and reconciled.generation_id == generation_id
                and reconciled.key == candidate.key
                and reconciled.storage_path == storage_path
                and reconciled.model_checksum == staged.metadata.checksum
                and reconciled.metadata == staged.metadata
            ):
                activated = True
            else:
                return self._remote_failure(
                    "remote_activation_failed", champion, remote_decision
                )
        if not activated:
            return self._remote_failure(
                "remote_head_conflict", champion, remote_decision
            )
        try:
            active_metadata, prune_audit_persisted = super()._publish_locked(
                staged,
                status=status,
                serialized_model=artifact,
            )
            if active_metadata != staged.metadata:
                raise ValueError("local cache metadata changed after activation")
        except Exception:
            self._mark_remote_activation_for_hydration(
                candidate.key,
                generation_id,
            )
            return remote_decision
        self._remember_activation(candidate.key, generation_id)
        return replace(
            remote_decision,
            audit_persisted=prune_audit_persisted,
        )

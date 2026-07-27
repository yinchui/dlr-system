import hashlib
import sys
from types import SimpleNamespace

import pytest

import modules.supabase_model_registry as supabase_registry_module
from modules.model_registry import (
    ModelAttempt,
    ModelCompatibility,
    ModelKey,
    ModelMetadata,
)
from modules.supabase_model_registry import (
    RemoteGeneration,
    SupabaseModelStore,
)


GENERATION_ID = "11111111-1111-4111-8111-111111111111"
EXPECTED_GENERATION_ID = "22222222-2222-4222-8222-222222222222"
CHECKSUM = "a" * 64


def _metadata(key, *, checksum=CHECKSUM, status="active"):
    return ModelMetadata(
        key=key,
        model_version="version-1",
        feature_columns=("wind_speed_local", "lag_1"),
        training_params={"max_depth": 3},
        random_seed=42,
        time_start="2025-01-01T00:00:00+08:00",
        time_end="2025-01-02T00:00:00+08:00",
        sample_count=4,
        evaluation_mode="temporal_holdout",
        metrics={
            "baseline_mae": 2.0,
            "baseline_rmse": 2.5,
            "corrected_mae": 1.0,
            "corrected_rmse": 1.5,
        },
        full_fit_metrics=None,
        residual_bounds=(-2.0, 2.0),
        input_data_hash="input-a",
        evaluation_set_hash="evaluation-a",
        compatibility=ModelCompatibility(
            dem_hash="dem-a",
            crs_hash="crs-a",
            coordinate_hash="coordinates-a",
            conductor_hash="conductor-a",
            feature_version="features-a",
            correction_config_hash="correction-a",
        ),
        dependency_versions={"python": "3.11", "joblib": "1.5.3"},
        cadence_minutes=30.0,
        training_contract_hash="c" * 64,
        backend_id="xgboost-residual-v1",
        training_outcome="data_fallback",
        checksum=checksum,
        status=status,
    )


def _generation_row(key, metadata, **overrides):
    path = (
        f"{key.project_id}/{key.line_id}/{key.tower_id}/{key.target}/"
        f"{GENERATION_ID}/model.joblib"
    )
    row = {
        "generation_id": GENERATION_ID,
        "project_id": key.project_id,
        "line_id": key.line_id,
        "tower_id": key.tower_id,
        "target": key.target,
        "model_version": metadata.model_version,
        "storage_path": path,
        "model_checksum": metadata.checksum,
        "metadata": metadata.to_dict(),
        "status": metadata.status,
    }
    row.update(overrides)
    return row


def _head_row(key, *, revision=1, **overrides):
    row = {
        "generation_id": GENERATION_ID,
        "revision": revision,
        "project_id": key.project_id,
        "line_id": key.line_id,
        "tower_id": key.tower_id,
        "target": key.target,
    }
    row.update(overrides)
    return row


class _Request:
    def __init__(self, client, kind, name, payload=None):
        self.client = client
        self.kind = kind
        self.name = name
        self.payload = payload
        self.operations = []

    def select(self, columns):
        self.operations.append(("select", columns))
        return self

    def match(self, values):
        self.operations.append(("match", dict(values)))
        return self

    def eq(self, column, value):
        self.operations.append(("eq", column, value))
        return self

    def upsert(self, values, **options):
        self.operations.append(("upsert", dict(values), dict(options)))
        return self

    def execute(self):
        self.client.requests.append(self)
        response = self.client.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(data=response)


class _Bucket:
    def __init__(self, client, bucket):
        self.client = client
        self.bucket = bucket

    def download(self, path):
        self.client.storage_calls.append(("download", self.bucket, path))
        result = self.client.storage_responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def upload(self, path, artifact, **options):
        self.client.storage_calls.append(
            ("upload", self.bucket, path, artifact, dict(options))
        )
        result = self.client.storage_responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _Storage:
    def __init__(self, client):
        self.client = client

    def from_(self, bucket):
        return _Bucket(self.client, bucket)


class _Client:
    def __init__(self, *, responses=(), storage_responses=()):
        self.responses = list(responses)
        self.storage_responses = list(storage_responses)
        self.requests = []
        self.storage_calls = []
        self.storage = _Storage(self)

    def table(self, name):
        return _Request(self, "table", name)

    def rpc(self, name, payload):
        return _Request(self, "rpc", name, dict(payload))


class _RequestBuildingFailureClient:
    def __init__(self, secret):
        self.secret = secret

    def table(self, name):
        raise RuntimeError(f"table setup failed with {self.secret}")

    def rpc(self, name, payload):
        raise RuntimeError(f"rpc setup failed with {self.secret}")


class _StorageApiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _attempt(key):
    return ModelAttempt(
        key=key,
        input_data_hash="i" * 64,
        evaluation_set_hash="e" * 64,
        policy_version="weather-promotion-v1",
        min_mae_improvement=0.0,
        training_contract_hash="c" * 64,
        backend_id="xgboost-residual-v1",
        feature_version="features-a",
    )


def test_store_loads_and_validates_current_generation():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    client = _Client(
        responses=[
            [_head_row(key, revision=3)],
            [_generation_row(key, metadata)],
        ]
    )

    generation = SupabaseModelStore(client).current(key)

    assert generation == RemoteGeneration(
        generation_id=GENERATION_ID,
        key=key,
        model_version="version-1",
        storage_path=(
            "project-a/line-a/001/wind_speed/"
            f"{GENERATION_ID}/model.joblib"
        ),
        model_checksum=CHECKSUM,
        metadata=metadata,
        status="active",
        revision=3,
    )
    assert client.requests[0].name == "dlr_model_heads"
    assert client.requests[0].operations[-1] == (
        "match",
        {
            "project_id": "project-a",
            "line_id": "line-a",
            "tower_id": "001",
            "target": "wind_speed",
        },
    )


def test_store_returns_none_only_for_an_empty_head_result():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    assert SupabaseModelStore(_Client(responses=[[]])).current(key) is None


def test_store_rejects_head_scope_that_differs_from_requested_key():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    client = _Client(
        responses=[
            [_head_row(key, project_id="project-b")],
            [_generation_row(key, metadata)],
        ]
    )

    with pytest.raises(OSError, match="head scope"):
        SupabaseModelStore(client).current(key)


@pytest.mark.parametrize(
    "head_rows,generation_rows,error",
    [
        (
            [
                {"generation_id": GENERATION_ID, "revision": 1},
                {"generation_id": EXPECTED_GENERATION_ID, "revision": 2},
            ],
            [],
            "head",
        ),
        ([{"generation_id": "not-a-uuid", "revision": 1}], [], "uuid"),
        ([{"generation_id": GENERATION_ID, "revision": 0}], [], "revision"),
    ],
)
def test_store_rejects_malformed_head_rows(head_rows, generation_rows, error):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    client = _Client(responses=[head_rows, generation_rows])

    with pytest.raises(OSError, match=error):
        SupabaseModelStore(client).current(key)


def test_store_rejects_generation_scope_or_path_mismatch():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    row = _generation_row(
        key,
        metadata,
        line_id="line-b",
        storage_path="other/model.joblib",
    )
    client = _Client(
        responses=[
            [_head_row(key)],
            [row],
        ]
    )

    with pytest.raises(OSError, match="scope|path"):
        SupabaseModelStore(client).current(key)


def test_store_rejects_duplicate_generation_rows():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    row = _generation_row(key, metadata)
    client = _Client(
        responses=[
            [_head_row(key)],
            [row, row],
        ]
    )

    with pytest.raises(OSError, match="generation response"):
        SupabaseModelStore(client).current(key)


def test_store_rejects_generation_checksum_that_differs_from_metadata():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    row = _generation_row(key, metadata, model_checksum="b" * 64)
    client = _Client(
        responses=[
            [_head_row(key)],
            [row],
        ]
    )

    with pytest.raises(OSError):
        SupabaseModelStore(client).current(key)


def test_store_rejects_candidate_status_for_remote_generation():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key, status="candidate")
    client = _Client(
        responses=[
            [_head_row(key)],
            [_generation_row(key, metadata)],
        ]
    )

    with pytest.raises(OSError):
        SupabaseModelStore(client).current(key)


def test_store_download_requires_bytes_and_matching_sha256():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    artifact = b"sealed model"
    checksum = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(key, checksum=checksum)
    generation = RemoteGeneration(
        generation_id=GENERATION_ID,
        key=key,
        model_version=metadata.model_version,
        storage_path=(
            "project-a/line-a/001/wind_speed/"
            f"{GENERATION_ID}/model.joblib"
        ),
        model_checksum=checksum,
        metadata=metadata,
        status=metadata.status,
        revision=1,
    )
    client = _Client(storage_responses=[artifact, b"damaged"])
    store = SupabaseModelStore(client)

    assert store.download(generation) == artifact
    with pytest.raises(OSError, match="checksum") as error:
        store.download(generation)

    assert type(error.value) is OSError


def test_store_treats_missing_download_object_as_per_key_corruption():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    artifact = b"sealed model"
    checksum = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(key, checksum=checksum)
    generation = RemoteGeneration(
        generation_id=GENERATION_ID,
        key=key,
        model_version=metadata.model_version,
        storage_path=(
            "project-a/line-a/001/wind_speed/"
            f"{GENERATION_ID}/model.joblib"
        ),
        model_checksum=checksum,
        metadata=metadata,
        status=metadata.status,
        revision=1,
    )
    secret = "sb_secret_do_not_expose"
    store = SupabaseModelStore(
        _Client(
            storage_responses=[
                _StorageApiError(404, f"object missing with {secret}")
            ]
        )
    )

    with pytest.raises(OSError, match="object is missing") as error:
        store.download(generation)

    assert type(error.value) is OSError
    assert secret not in str(error.value)


@pytest.mark.parametrize("status", [None, 401, 403, 429, 500])
def test_store_treats_systemic_download_failures_as_transport(status):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    artifact = b"sealed model"
    checksum = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(key, checksum=checksum)
    generation = RemoteGeneration(
        generation_id=GENERATION_ID,
        key=key,
        model_version=metadata.model_version,
        storage_path=(
            "project-a/line-a/001/wind_speed/"
            f"{GENERATION_ID}/model.joblib"
        ),
        model_checksum=checksum,
        metadata=metadata,
        status=metadata.status,
        revision=1,
    )

    with pytest.raises(OSError) as error:
        SupabaseModelStore(
            _Client(
                storage_responses=[
                    _StorageApiError(status, "systemic failure")
                ]
            )
        ).download(generation)

    assert type(error.value) is getattr(
        supabase_registry_module,
        "SupabaseTransportError",
        None,
    )


def test_store_uploads_to_immutable_binary_generation_path():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    artifact = b"sealed model"
    expected_path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )
    client = _Client(
        storage_responses=[
            {"path": expected_path},
            SimpleNamespace(path=expected_path),
        ]
    )
    store = SupabaseModelStore(client)

    path = store.upload(GENERATION_ID, key, artifact)
    repeated_path = store.upload(GENERATION_ID, key, artifact)

    assert path == expected_path
    assert repeated_path == path
    expected_call = (
        "upload",
        "dlr-models",
        path,
        artifact,
        {
            "file_options": {
                "content-type": "application/octet-stream",
                "upsert": "true",
            }
        },
    )
    assert client.storage_calls == [expected_call, expected_call]


def test_store_uses_distinct_paths_for_distinct_generation_uuids():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    other_generation_id = "33333333-3333-4333-8333-333333333333"
    first_path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )
    second_path = (
        "project-a/line-a/001/wind_speed/"
        f"{other_generation_id}/model.joblib"
    )
    client = _Client(
        storage_responses=[{"path": first_path}, {"path": second_path}]
    )
    store = SupabaseModelStore(client)

    first_result = store.upload(GENERATION_ID, key, b"first")
    second_result = store.upload(other_generation_id, key, b"second")

    assert first_result == first_path
    assert second_result == second_path
    assert client.storage_calls == [
        (
            "upload",
            "dlr-models",
            first_path,
            b"first",
            {
                "file_options": {
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                }
            },
        ),
        (
            "upload",
            "dlr-models",
            second_path,
            b"second",
            {
                "file_options": {
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                }
            },
        ),
    ]


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"path": "wrong/model.joblib"},
        SimpleNamespace(path="wrong/model.joblib"),
        object(),
    ],
)
def test_store_rejects_invalid_upload_responses(response):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    store = SupabaseModelStore(_Client(storage_responses=[response]))

    with pytest.raises(OSError, match="upload response") as error:
        store.upload(GENERATION_ID, key, b"sealed model")

    assert type(error.value) is OSError


def test_store_activation_passes_complete_metadata_and_expected_head():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    client = _Client(responses=[True])
    store = SupabaseModelStore(client)
    path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )

    activated = store.activate(
        GENERATION_ID,
        key,
        metadata,
        path,
        expected_generation_id=EXPECTED_GENERATION_ID,
    )

    assert activated is True
    request = client.requests[0]
    assert request.kind == "rpc"
    assert request.name == "activate_dlr_model_generation"
    assert request.payload == {
        "p_generation_id": GENERATION_ID,
        "p_project_id": "project-a",
        "p_line_id": "line-a",
        "p_tower_id": "001",
        "p_target": "wind_speed",
        "p_model_version": "version-1",
        "p_storage_path": path,
        "p_model_checksum": CHECKSUM,
        "p_metadata": metadata.to_dict(),
        "p_status": "active",
        "p_expected_generation_id": EXPECTED_GENERATION_ID,
    }


@pytest.mark.parametrize("response", [None, [], 1, "true"])
def test_store_rejects_malformed_activation_responses(response):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )

    with pytest.raises(OSError, match="activation response") as error:
        SupabaseModelStore(_Client(responses=[response])).activate(
            GENERATION_ID,
            key,
            metadata,
            path,
            expected_generation_id=None,
        )

    assert type(error.value) is OSError


def test_store_rejection_lookup_and_upsert_are_fully_scoped():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    attempt = _attempt(key)
    client = _Client(
        responses=[
            [
                {
                    "project_id": "project-a",
                    "line_id": "line-a",
                    "tower_id": "001",
                    "target": "wind_speed",
                    "attempt_fingerprint": attempt.fingerprint,
                }
            ],
            [
                {
                    "project_id": "project-a",
                    "line_id": "line-a",
                    "tower_id": "001",
                    "target": "wind_speed",
                    "attempt_fingerprint": attempt.fingerprint,
                }
            ],
        ]
    )
    store = SupabaseModelStore(client)

    assert store.was_rejected(attempt) is True
    store.record_rejection(attempt, "insufficient_mae_improvement")

    lookup = client.requests[0]
    assert lookup.operations[-1] == (
        "match",
        {
            "project_id": "project-a",
            "line_id": "line-a",
            "tower_id": "001",
            "target": "wind_speed",
            "attempt_fingerprint": attempt.fingerprint,
        },
    )
    upsert = client.requests[1]
    operation, values, options = upsert.operations[-1]
    assert operation == "upsert"
    assert values == {
        "project_id": "project-a",
        "line_id": "line-a",
        "tower_id": "001",
        "target": "wind_speed",
        "attempt_fingerprint": attempt.fingerprint,
        "champion_context_hash": None,
        "reason": "insufficient_mae_improvement",
        "attempt": attempt.to_dict(),
    }
    assert options == {
        "on_conflict": (
            "project_id,line_id,tower_id,target,attempt_fingerprint"
        )
    }


def test_store_rejects_duplicate_rejection_write_rows():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    attempt = _attempt(key)
    row = {
        **key.__dict__,
        "attempt_fingerprint": attempt.fingerprint,
    }
    client = _Client(responses=[[row, row]])

    with pytest.raises(OSError, match="rejection write"):
        SupabaseModelStore(client).record_rejection(attempt, "reason")


@pytest.mark.parametrize(
    "response",
    [None, {}, [{}, {}], [{"attempt_fingerprint": "wrong"}]],
)
def test_store_rejects_malformed_rejection_lookup_responses(response):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    with pytest.raises(OSError, match="rejection"):
        SupabaseModelStore(_Client(responses=[response])).was_rejected(
            _attempt(key)
        )


def test_store_rejects_rejection_lookup_scope_mismatch():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    attempt = _attempt(key)
    response = [
        {
            "project_id": "project-a",
            "line_id": "line-b",
            "tower_id": "001",
            "target": "wind_speed",
            "attempt_fingerprint": attempt.fingerprint,
        }
    ]

    with pytest.raises(OSError, match="rejection"):
        SupabaseModelStore(_Client(responses=[response])).was_rejected(attempt)


def test_store_translates_sdk_failures_without_leaking_secret_values():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    secret = "sb_secret_do_not_expose"
    client = _Client(responses=[RuntimeError(f"request failed with {secret}")])

    with pytest.raises(OSError) as error:
        SupabaseModelStore(client).current(key)

    assert type(error.value) is getattr(
        supabase_registry_module,
        "SupabaseTransportError",
        None,
    )
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    "operation",
    ["current", "activate", "was_rejected", "record_rejection"],
)
def test_store_translates_request_building_failures_without_leaking_secrets(
    operation,
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    metadata = _metadata(key)
    attempt = _attempt(key)
    path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )
    secret = "sb_secret_do_not_expose"
    store = SupabaseModelStore(_RequestBuildingFailureClient(secret))

    calls = {
        "current": lambda: store.current(key),
        "activate": lambda: store.activate(
            GENERATION_ID,
            key,
            metadata,
            path,
            expected_generation_id=None,
        ),
        "was_rejected": lambda: store.was_rejected(attempt),
        "record_rejection": lambda: store.record_rejection(attempt, "reason"),
    }

    with pytest.raises(OSError) as error:
        calls[operation]()

    assert type(error.value) is getattr(
        supabase_registry_module,
        "SupabaseTransportError",
        None,
    )
    assert secret not in str(error.value)


@pytest.mark.parametrize("operation", ["download", "upload"])
def test_store_translates_storage_failures_without_leaking_secrets(operation):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    artifact = b"sealed model"
    checksum = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(key, checksum=checksum)
    generation = RemoteGeneration(
        generation_id=GENERATION_ID,
        key=key,
        model_version=metadata.model_version,
        storage_path=(
            "project-a/line-a/001/wind_speed/"
            f"{GENERATION_ID}/model.joblib"
        ),
        model_checksum=checksum,
        metadata=metadata,
        status=metadata.status,
        revision=1,
    )
    secret = "sb_secret_do_not_expose"
    store = SupabaseModelStore(
        _Client(storage_responses=[RuntimeError(f"failed with {secret}")])
    )

    calls = {
        "download": lambda: store.download(generation),
        "upload": lambda: store.upload(GENERATION_ID, key, artifact),
    }

    with pytest.raises(OSError) as error:
        calls[operation]()

    assert type(error.value) is getattr(
        supabase_registry_module,
        "SupabaseTransportError",
        None,
    )
    assert secret not in str(error.value)


def test_store_lazily_creates_client_without_retaining_raw_credentials(
    monkeypatch,
):
    client = _Client()
    captured = {}

    class UnitClientOptions:
        def __init__(self, **values):
            self.values = values

    def create_client(url, secret_key, *, options):
        captured.update(
            url=url,
            secret_key=secret_key,
            options=options,
        )
        return client

    monkeypatch.setitem(
        sys.modules,
        "supabase",
        SimpleNamespace(
            ClientOptions=UnitClientOptions,
            create_client=create_client,
        ),
    )
    secret = "sb_secret_do_not_retain"

    store = SupabaseModelStore.from_credentials(
        " https://project.supabase.co ",
        f" {secret} ",
    )

    assert captured == {
        "url": "https://project.supabase.co",
        "secret_key": secret,
        "options": captured["options"],
    }
    assert isinstance(captured["options"], UnitClientOptions)
    assert captured["options"].values == {
        "postgrest_client_timeout": 5.0,
        "storage_client_timeout": 5.0,
        "function_client_timeout": 5.0,
    }
    assert vars(store) == {"_client": client, "bucket": "dlr-models"}
    assert secret not in repr(vars(store))

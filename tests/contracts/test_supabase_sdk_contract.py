import hashlib
import inspect
from importlib.metadata import PackageNotFoundError, version

import pytest

from modules.model_registry import ModelCompatibility, ModelKey, ModelMetadata
from modules.supabase_model_registry import RemoteGeneration, SupabaseModelStore

try:
    for distribution in ("supabase", "storage3", "postgrest"):
        version(distribution)
except PackageNotFoundError:
    REAL_SDK_INSTALLED = False
else:
    REAL_SDK_INSTALLED = True


if REAL_SDK_INSTALLED:
    from postgrest.base_request_builder import APIResponse
    from storage3.exceptions import StorageApiError
    from storage3.types import UploadResponse
    from supabase import ClientOptions, create_client
else:
    APIResponse = None
    StorageApiError = None
    UploadResponse = None
    ClientOptions = None
    create_client = None


pytestmark = pytest.mark.skipif(
    not REAL_SDK_INSTALLED,
    reason="real supabase-py, storage3, and postgrest packages are not installed",
)


GENERATION_ID = "11111111-1111-4111-8111-111111111111"


class _ExecuteRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class _RpcBoundaryClient:
    def __init__(self, response):
        self.response = response

    def rpc(self, _name, _payload):
        return _ExecuteRequest(self.response)


class _UploadBucket:
    def __init__(self, response):
        self.response = response

    def upload(self, _path, _artifact, **_options):
        return self.response


class _UploadStorage:
    def __init__(self, response):
        self.response = response

    def from_(self, _bucket):
        return _UploadBucket(self.response)


class _UploadBoundaryClient:
    def __init__(self, response):
        self.storage = _UploadStorage(response)


class _DownloadBucket:
    def __init__(self, error):
        self.error = error

    def download(self, _path):
        raise self.error


class _DownloadStorage:
    def __init__(self, error):
        self.error = error

    def from_(self, _bucket):
        return _DownloadBucket(self.error)


class _DownloadBoundaryClient:
    def __init__(self, error):
        self.storage = _DownloadStorage(error)


def _metadata(key: ModelKey, checksum: str) -> ModelMetadata:
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
        status="active",
    )


def test_real_client_options_support_all_bounded_service_timeouts():
    options = ClientOptions(
        postgrest_client_timeout=5.0,
        storage_client_timeout=5.0,
        function_client_timeout=5.0,
    )

    assert options.postgrest_client_timeout == 5.0
    assert options.storage_client_timeout == 5.0
    assert options.function_client_timeout == 5.0
    inspect.signature(create_client).bind(
        "https://ciapxhuldarsupmvrgwu.supabase.co",
        "contract-test-key",
        options=options,
    )


def test_real_sdk_builders_accept_all_production_call_shapes_without_io():
    store = SupabaseModelStore.from_credentials(
        "https://abcdefghijklmnopqrst.supabase.co",
        "contract-test-key",
    )
    client = store._client
    path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )
    scope = {
        "project_id": "project-a",
        "line_id": "line-a",
        "tower_id": "001",
        "target": "wind_speed",
    }

    bucket = client.storage.from_(store.bucket)
    inspect.signature(bucket.upload).bind(
        path,
        b"sealed model",
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "false",
        },
    )
    inspect.signature(bucket.download).bind(path)

    # Supabase request builders are lazy until execute() is called.
    builders = [
        client.rpc("activate_dlr_model_generation", {"p_target": "wind_speed"}),
        client.table("dlr_model_heads")
        .select("generation_id,revision,project_id,line_id,tower_id,target")
        .match(scope),
        client.table("dlr_model_generations")
        .select("generation_id,storage_path")
        .eq("generation_id", GENERATION_ID),
        client.table("dlr_model_rejections")
        .select("project_id,line_id,tower_id,target,attempt_fingerprint")
        .match({**scope, "attempt_fingerprint": "a" * 64}),
        client.table("dlr_model_rejections").upsert(
            {**scope, "attempt_fingerprint": "a" * 64},
            on_conflict=(
                "project_id,line_id,tower_id,target,attempt_fingerprint"
            ),
        ),
    ]

    for builder in builders:
        assert callable(builder.execute)
        inspect.signature(builder.execute).bind()


def test_real_storage_upload_response_exposes_canonical_path():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )
    response = UploadResponse(path=path, fullPath=path)
    store = SupabaseModelStore(_UploadBoundaryClient(response))

    assert store.upload(GENERATION_ID, key, b"sealed model") == path


def test_real_postgrest_response_preserves_rpc_boolean_scalar():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    artifact = b"sealed model"
    checksum = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(key, checksum)
    path = (
        "project-a/line-a/001/wind_speed/"
        f"{GENERATION_ID}/model.joblib"
    )
    response = APIResponse[bool](data=True)
    store = SupabaseModelStore(_RpcBoundaryClient(response))

    assert store.activate(
        GENERATION_ID,
        key,
        metadata,
        path,
        expected_generation_id=None,
    ) is True


def test_real_storage_missing_object_error_remains_per_key_corruption():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    artifact = b"sealed model"
    checksum = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(key, checksum)
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
    error = StorageApiError("Object not found", "not_found", 404)
    store = SupabaseModelStore(_DownloadBoundaryClient(error))

    with pytest.raises(OSError, match="object is missing") as caught:
        store.download(generation)

    assert type(caught.value) is OSError

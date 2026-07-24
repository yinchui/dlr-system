# Sealed Production Training Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace arbitrary Python trainer/factory persistence with one sealed, deterministic XGBoost production backend while preserving physical DLR fallback, small-sample training, weather-only evaluation, model reuse, and per-tower isolation.

**Architecture:** A frozen `SealedEstimatorSpec` becomes the only authority for production model training and reuse. The pipeline may still receive injected trainers for compatibility, but only the default sealed `ResidualTrainer` can publish models or deterministic rejection records; model loading is also scoped by the current sealed contract. Existing callable reflection code is removed after all production paths consume the sealed spec.

**Tech Stack:** Python 3.11, pandas, NumPy, XGBoost 3.x, joblib, filelock, pytest, Ruff.

---

## Preconditions

- Worktree: `/Users/aa/.config/superpowers/worktrees/12.24/dlr-correction-sag-validation-worktree`
- Branch: `feature/dlr-correction-sag-validation`
- Starting HEAD includes design commit `b494c82`.
- Python: `/opt/homebrew/opt/python@3.11/bin/python3.11`
- Read first: `docs/plans/2026-07-24-sealed-training-backend-design.md`
- Follow `@test-driven-development`, `@systematic-debugging`, and `@verification-before-completion`.
- Only one implementation agent may modify shared code at a time.

### Task 1: Add the sealed XGBoost spec and estimator attestation

**Files:**
- Modify: `modules/ai_training.py:105-260`
- Modify: `modules/ai_training.py:1389-1637`
- Test: `tests/modules/test_ai_training.py`
- Create: `tests/integration/test_sealed_xgboost_lifecycle.py`

**Step 1: Add a real-backend characterization and failing spec tests**

First add a characterization test that uses no injected trainer or estimator:

```python
def test_real_xgboost_trains_persists_and_reuses_per_tower_models(tmp_path):
    first = DlrPipeline(model_root=tmp_path).run(
        physical=_weather_with_independent_segments("physical"),
        truth=_weather_with_independent_segments("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )
    second = DlrPipeline(model_root=tmp_path).run(
        physical=_later_weather("physical"),
        truth=None,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert first.model_report.trained_targets
    assert second.model_report.loaded_targets == first.model_report.trained_targets
    assert second.model_report.used_targets == first.model_report.trained_targets
    assert np.isfinite(first.max_currents).all()
    assert np.isfinite(second.max_currents).all()
```

This captures the production workflow before the contract refactor. Then add tests that define the new production API:

```python
def test_default_trainer_has_sealed_xgboost_spec():
    trainer = ResidualTrainer()
    spec = trainer.sealed_estimator_spec

    assert spec.backend_id == "xgboost-residual-v1"
    assert spec.estimator_path == "xgboost.XGBRegressor"
    assert spec.parameters["random_state"] == 42
    assert spec.parameters["n_jobs"] == 1
    assert spec.distributions["xgboost"] == importlib.metadata.version("xgboost")
    assert len(spec.implementation_sha256) == 64


def test_custom_factory_is_not_production_eligible():
    trainer = ResidualTrainer(estimator_factory=MeanResidualEstimator)

    assert trainer.production_eligible is False
    assert trainer.sealed_estimator_spec is None


def test_estimator_attestation_rejects_parameter_or_seed_mismatch(monkeypatch):
    trainer = ResidualTrainer()
    estimator = default_estimator()
    estimator.set_params(random_state=None)

    with pytest.raises(TrainingContractError, match="attestation|random_state"):
        trainer.attest_estimator(estimator)
```

Also cover:

- type mismatch;
- missing/ambiguous distribution mapping;
- implementation file hash change;
- actual estimator parameters exactly equal the frozen spec;
- `ConstantResidualEstimator` is an internal deterministic data fallback, not a sealed XGBoost estimator.

**Step 2: Verify the characterization passes and the new API is RED**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/integration/test_sealed_xgboost_lifecycle.py -q

/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_ai_training.py \
  -q -k 'sealed_xgboost_spec or production_eligible or estimator_attestation'
```

Expected: the real XGBoost characterization PASS; the new API tests FAIL because `SealedEstimatorSpec`, `sealed_estimator_spec`, `production_eligible`, and `attest_estimator()` do not exist.

**Step 3: Implement the frozen spec**

Add a frozen dataclass with immutable values:

```python
@dataclass(frozen=True)
class SealedEstimatorSpec:
    schema_version: int
    backend_id: str
    estimator_path: str
    parameters_json: str
    random_seed: int
    distributions: tuple[tuple[str, str], ...]
    implementation_sha256: str
    policy_version: str

    @property
    def parameters(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.parameters_json))

    def digest(self) -> str:
        return _stable_json_hash(asdict(self))
```

Implement only the default production builder:

```python
def sealed_xgboost_spec() -> SealedEstimatorSpec:
    estimator_type = _load_xgb_regressor()
    estimator = estimator_type(**_DEFAULT_ESTIMATOR_PARAMETERS)
    distributions = _resolved_distribution_versions(estimator_type.__module__)
    implementation_path = Path(inspect.getfile(estimator_type)).resolve(strict=True)
    return SealedEstimatorSpec(
        schema_version=1,
        backend_id="xgboost-residual-v1",
        estimator_path=_qualified_name(estimator_type),
        parameters_json=_canonical_json(estimator.get_params(deep=False)),
        random_seed=42,
        distributions=tuple(sorted(distributions.items())),
        implementation_sha256=_sha256_path(implementation_path),
        policy_version="weather-residual-training-v1",
    )
```

Use `importlib.metadata.packages_distributions()` to map import roots to real distribution names. Reject missing, ambiguous, or unreadable versions with `TrainingContractError`.

Update `ResidualTrainer.__init__`:

- default factory: build and retain the sealed spec, `production_eligible=True`;
- injected factory: `sealed_estimator_spec=None`, `production_eligible=False`;
- do not infer production eligibility from arbitrary descriptors.

Add `attest_estimator()` and call it immediately after every default estimator construction and before `fit()`. It must compare exact type, exact `get_params(deep=False)`, seed, distribution versions, and implementation hash.

**Step 4: Run targeted GREEN tests**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/modules/test_ai_training.py -q
```

Expected: all training tests PASS. Custom estimator unit tests may continue to train, but their trainer is explicitly non-production.

**Step 5: Commit**

```bash
git add modules/ai_training.py tests/modules/test_ai_training.py \
  tests/integration/test_sealed_xgboost_lifecycle.py
git commit -m "feat: seal the production XGBoost estimator"
```

### Task 2: Enforce the sealed backend at the DLR pipeline boundary

**Files:**
- Modify: `modules/dlr_pipeline.py:890-930`
- Modify: `modules/dlr_pipeline.py:1119-1560`
- Modify: `tests/integration/test_dlr_pipeline.py`

**Step 1: Write failing pipeline boundary tests**

Add tests:

```python
def test_custom_trainer_cannot_publish_or_enable_ai(tmp_path):
    trainer = _CountingTrainer()
    result = DlrPipeline(model_root=tmp_path, trainer=trainer).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert trainer.calls == []
    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert {item.reason for item in result.model_report.fallbacks} == {
        "unsupported_training_backend"
    }
    assert np.isfinite(result.max_currents).all()
    assert list(tmp_path.iterdir()) == []


def test_custom_estimator_factory_cannot_enter_persistent_pipeline(tmp_path):
    trainer = ResidualTrainer(estimator_factory=_LinearResidualRegressor)
    result = DlrPipeline(model_root=tmp_path, trainer=trainer).run(...)

    assert result.model_report.trained_targets == ()
    assert all(
        fallback.reason == "unsupported_training_backend"
        for fallback in result.model_report.fallbacks
    )
```

Cover both `model_persistence_allowed=True` and `False`. In both modes, an unsealed backend must not become an AI source; physical DLR must continue.

**Step 2: Run tests and verify RED**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/integration/test_dlr_pipeline.py \
  -q -k 'custom_trainer_cannot or custom_estimator_factory_cannot'
```

Expected: FAIL because current explicit descriptors still allow custom trainers to train and publish.

**Step 3: Implement one eligibility check**

Add one narrow helper:

```python
def _sealed_trainer_for_interval(self, interval_minutes: float) -> ResidualTrainer:
    trainer = self._trainer_for_interval(interval_minutes)
    if type(trainer) is not ResidualTrainer or not trainer.production_eligible:
        raise TrainingContractError("unsupported_training_backend")
    return trainer
```

Use it only when training is actually needed. Model loading may still occur before training, but Task 3 will bind load compatibility to the current sealed spec.

Handle `TrainingContractError` per key and append one authoritative `ModelFallback(key, "unsupported_training_backend")`. Do not construct a registry solely for an unsealed, nonpersistent path. Do not catch thermal calculation errors inside this boundary.

Refactor integration tests that currently use `_CountingTrainer`, `_PoorTemporalTrainer`, or `_AlwaysPoorTrainer` to assert lifecycle through real registry/report state. Keep custom trainers only in tests whose expected result is operational fallback.

**Step 4: Run integration GREEN tests**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/integration/test_dlr_pipeline.py \
  tests/modules/test_ai_training.py -q
```

Expected: PASS. No custom injected trainer is published or cached.

**Step 5: Commit**

```bash
git add modules/dlr_pipeline.py tests/integration/test_dlr_pipeline.py
git commit -m "fix: restrict DLR persistence to the sealed backend"
```

### Task 3: Scope model loading to the current sealed contract

**Files:**
- Modify: `modules/model_registry.py:219-417`
- Modify: `modules/model_registry.py:1213-1362`
- Modify: `modules/dlr_pipeline.py:1150-1410`
- Test: `tests/modules/test_model_registry.py`
- Test: `tests/integration/test_dlr_pipeline.py`

**Step 1: Write failing load-compatibility tests**

Add tests:

```python
def test_load_rejects_model_from_a_different_sealed_backend_contract(tmp_path):
    registry = ModelRegistry(tmp_path)
    candidate = model_candidate(
        ModelKey("project-a", "line-a", "001", "wind_speed"),
        training_contract_hash="a" * 64,
        backend_id="xgboost-residual-v1",
    )
    assert registry.promote(candidate).promoted

    loaded = registry.load(
        candidate.key,
        expected_compatibility=compatible_hashes(),
        expected_training_contract_hash="b" * 64,
        expected_backend_id="xgboost-residual-v1",
    )

    assert loaded.bundle is None
    assert loaded.fallback_reason == "incompatible_training_contract_hash"


def test_legacy_model_isolated_when_current_backend_contract_is_required(tmp_path):
    ...
    assert loaded.fallback_reason == "incompatible_training_contract_hash"
```

Also cover backend id mismatch and dependency-version-induced contract changes.

**Step 2: Run tests and verify RED**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_model_registry.py \
  -q -k 'sealed_backend_contract or legacy_model_isolated'
```

Expected: FAIL because `load()` currently does not accept or validate a runtime training contract.

**Step 3: Extend metadata and load validation**

Add `backend_id` to new `ModelMetadata`. It is required for new candidates; `from_dict()` assigns a legacy backend sentinel only when the field is absent. Explicit legacy sentinels remain rejected for newly serialized payloads.

Extend `load()` and `load_many()` with required production expectations:

```python
def load(
    self,
    key: ModelKey,
    *,
    expected_compatibility: ModelCompatibility,
    expected_training_contract_hash: str,
    expected_backend_id: str,
) -> ModelLoadResult:
    ...
```

Inside `_read_generation()`, reject only the affected key when either expected value differs. Preserve checksum, path, permission, and schema validation ordering.

Update `candidate_from_training_result()` to copy the sealed backend id from the result metadata and verify result/bundle/metadata agree.

Update pipeline loads to derive the target/cadence scoped contract from the default sealed trainer before `load_many()`. If sealed spec construction fails, treat every requested model as operationally unavailable and continue physical DLR.

**Step 4: Run registry and integration tests**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_model_registry.py \
  tests/integration/test_dlr_pipeline.py -q
```

Expected: PASS, including old metadata migration, corrupt-model isolation, concurrent promotion, and model reuse.

**Step 5: Commit**

```bash
git add modules/model_registry.py modules/dlr_pipeline.py \
  tests/modules/test_model_registry.py tests/integration/test_dlr_pipeline.py
git commit -m "fix: invalidate models outside the sealed contract"
```

### Task 4: Remove arbitrary callable reflection from production contracts

**Files:**
- Modify: `modules/ai_training.py:105-1033`
- Modify: `tests/modules/test_ai_training.py:308-779`
- Modify: `tests/integration/test_dlr_pipeline.py`
- Modify: `tests/modules/test_model_registry.py`

**Step 1: Write failing regression tests for the removed capability**

Replace reflection-specific expectations with boundary tests:

```python
@pytest.mark.parametrize(
    "factory",
    [
        RandomForestRegressor,
        partial(RandomForestRegressor, random_state=42),
        ConfiguredEstimatorFactory(offset=1.0),
        lambda: FixedResidualEstimator(1.0),
    ],
)
def test_arbitrary_factory_never_claims_a_production_contract(factory):
    trainer = ResidualTrainer(estimator_factory=factory)

    assert trainer.production_eligible is False
    assert trainer.sealed_estimator_spec is None
```

Keep direct unit tests for custom estimators only where they exercise fitting, metrics, clipping, schema validation, or operational fallback. Remove tests whose sole purpose is proving arbitrary callable introspection.

**Step 2: Run the boundary tests before deletion**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_ai_training.py \
  -q -k 'arbitrary_factory_never_claims'
```

Expected: PASS only after Task 1's eligibility field; use this as the characterization guard before refactoring.

**Step 3: Delete the obsolete reflection engine**

Remove production-only introspection machinery, including:

- `_FactoryContractContext`;
- `FrozenCallableContract`;
- `_factory_state`, `_explicit_contract_descriptor`, and `_contract_value` when no longer used by metadata sanitation;
- `_code_contract`, `_referenced_global_dependencies`, module/global/function/class/callable contract walkers;
- arbitrary trainer runtime descriptors.

Retain the independent bounded metadata sanitizer. Do not merge estimator contract data with persisted training parameter summaries.

Make `TrainingContract` derive only from:

- `SealedEstimatorSpec.digest()` for production;
- feature columns, target, physical/truth columns, cadence, feature version, and training policy;
- a non-production sentinel for direct custom-estimator unit training that `candidate_from_training_result()` rejects.

The resulting code must not inspect arbitrary callable globals, private attributes, closure alias topology, or dynamic module access.

**Step 4: Run all Task 11 tests and inspect code size**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_ai_training.py \
  tests/modules/test_model_registry.py \
  tests/integration/test_dlr_pipeline.py -q

rg -n '_code_contract|_callable_contract|FrozenCallableContract|training_contract_descriptor' \
  modules/ai_training.py modules/dlr_pipeline.py
```

Expected: tests PASS; `rg` finds no obsolete arbitrary reflection API. `modules/ai_training.py` should materially shrink rather than replacing the walker with another generic walker.

**Step 5: Commit**

```bash
git add modules/ai_training.py modules/dlr_pipeline.py modules/model_registry.py \
  tests/modules/test_ai_training.py tests/modules/test_model_registry.py \
  tests/integration/test_dlr_pipeline.py
git commit -m "refactor: remove arbitrary training code reflection"
```

### Task 5: Verify the real XGBoost lifecycle and finish Task 11 review

**Files:**
- Modify: `docs/TESTING.md`
- Modify: `docs/plans/2026-07-23-dlr-correction-and-sag-validation-implementation.md`

**Step 1: Run the accumulated real-backend acceptance tests**

The real lifecycle characterization was added before Task 1. Tasks 2 and 3 must add their XGBoost ImportError, unsealed-backend, dependency invalidation, and normal rejection-cache tests before implementing those behaviors. Run the accumulated file now:

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/integration/test_sealed_xgboost_lifecycle.py -q
```

Expected: PASS, including training/reuse, weather-only metrics, small full-fit provisional persistence, operational retry, contract invalidation, and deterministic quality rejection caching.

**Step 2: Update testing documentation and the parent plan**

Document:

- production persistence supports only the sealed XGBoost backend;
- custom trainers/factories are non-production and cannot write models or rejection caches;
- current XGBoost version and deterministic parameters are verified at runtime;
- dependency/backend changes invalidate old models per key;
- Task 12 remains the next implementation task after final Task 11 approval.

Mark the Task 11 addendum complete in the parent implementation plan without marking Task 12 complete.

**Step 3: Run final verification**

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest -q
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_thermal_engine_ieee738.py \
  tests/modules/test_thermal_transient.py -q
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/integration -q
/opt/homebrew/opt/python@3.11/bin/python3.11 -m ruff check --target-version py311 \
  modules/ai_training.py modules/dlr_pipeline.py modules/model_registry.py \
  tests/modules/test_ai_training.py tests/modules/test_model_registry.py \
  tests/integration/test_dlr_pipeline.py \
  tests/integration/test_sealed_xgboost_lifecycle.py
/opt/homebrew/opt/python@3.11/bin/python3.11 -m py_compile \
  modules/*.py utils/*.py config/*.py dispatch_app_st.py thermal_functions.py
git diff --check
git status --short
```

Expected:

- all commands exit 0;
- full suite has zero failures;
- worktree contains only the intended documentation/test changes before commit;
- no UI file is modified by this addendum.

**Step 4: Commit**

```bash
git add docs/TESTING.md \
  docs/plans/2026-07-23-dlr-correction-and-sag-validation-implementation.md
git commit -m "test: verify the sealed XGBoost model lifecycle"
```

### Task 6: Run two-stage review before Task 12

**Files:**
- No production edits unless a reviewer identifies a verified issue.

**Step 1: Request specification review**

Review every requirement in `docs/plans/2026-07-24-sealed-training-backend-design.md` against the complete addendum commit range. Require the reviewer to end with `Spec compliant: Yes/No`.

**Step 2: Fix any specification gap with RED/GREEN**

If the answer is No, return the finding to the same implementation agent, add a failing regression test, fix it, and ask the same reviewer to re-review.

**Step 3: Request quality review**

Use the same Task 11 quality reviewer. Require independent probes for:

- arbitrary/custom trainers cannot publish or poison sidecars;
- actual estimator params and random seed match metadata;
- XGBoost/distribution changes invalidate load and attempt caches;
- operation failures remain retryable;
- no generic callable reflection remains;
- old model isolation and concurrent promotion remain correct.

The reviewer must end with `Ready to proceed: Yes/No`.

**Step 4: Proceed only after approval**

Task 11 is complete only when both reviewers return Yes and fresh full verification is green. Then update the active task plan and begin Task 12.

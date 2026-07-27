# Supabase XGBoost Model Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Persist sealed per-line/per-tower XGBoost weather-correction models in Supabase across Streamlit container restarts while preserving all current model validation, weather-only promotion, DLR fallback, and sag independence.

**Architecture:** A private Supabase Storage bucket holds immutable `model.joblib` generations, while PostgreSQL stores generation metadata, the current head per model scope, and deterministic rejection fingerprints. `SupabaseModelRegistry` subclasses the existing local `ModelRegistry`, using a temporary local registry for its proven validation and promotion logic and treating Supabase as the remote source of truth.

**Tech Stack:** Python 3.11, Supabase Python 2.x, PostgreSQL/PLpgSQL, Supabase Storage, XGBoost, joblib, pytest, Ruff, Streamlit Community Cloud.

---

## Preconditions

- Worktree: `/Users/aa/.config/superpowers/worktrees/12.24/dlr-correction-sag-validation-worktree`
- Branch: `feature/dlr-correction-sag-validation`
- Starting design commit: `9623a5a`
- Supabase project: `ciapxhuldarsupmvrgwu`
- Read first: `docs/plans/2026-07-28-supabase-model-persistence-design.md`
- Follow `@test-driven-development`, `@systematic-debugging`, `@requesting-code-review`, and `@verification-before-completion`.
- Only one implementation agent may edit shared code at a time.
- Never write Supabase keys to Git, test snapshots, shell history files, logs, or page output.

### Task 1: Define the Supabase database and Storage contract

**Files:**
- Create: `supabase/migrations/20260728000000_dlr_model_registry.sql`
- Create: `tests/config/test_supabase_migration.py`

**Step 1: Write failing migration contract tests**

Add tests that require the migration to:

- create private bucket `dlr-models` idempotently;
- create `dlr_model_generations`, `dlr_model_heads`, and `dlr_model_rejections`;
- use a four-column scope key and target/status/checksum constraints;
- enable RLS on all three public tables without anonymous policies;
- define `activate_dlr_model_generation` as `SECURITY DEFINER` with an empty `search_path`;
- compare the expected head generation before inserting and switching the head;
- revoke public execution and grant only `service_role`.

Run:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/config/test_supabase_migration.py -q
```

Expected: FAIL because the migration does not exist.

**Step 2: Add the idempotent SQL migration**

The activation function must accept one complete generation and return `true` only when the expected head still matches:

```sql
create or replace function public.activate_dlr_model_generation(
  p_generation_id uuid,
  p_project_id text,
  p_line_id text,
  p_tower_id text,
  p_target text,
  p_model_version text,
  p_storage_path text,
  p_model_checksum text,
  p_metadata jsonb,
  p_status text,
  p_expected_generation_id uuid default null
) returns boolean
language plpgsql
security definer
set search_path = ''
```

Within one transaction it must lock the scope head, fail closed on a CAS mismatch, insert the immutable generation, and upsert the head with `revision + 1`. Bucket creation must set `public=false` and allow only `application/octet-stream`.

**Step 3: Run GREEN tests and lint the SQL text contract**

Run the Task 1 test again and `git diff --check`.

**Step 4: Commit**

```bash
git add supabase/migrations/20260728000000_dlr_model_registry.sql \
  tests/config/test_supabase_migration.py
git commit -m "feat: define Supabase model registry schema"
```

### Task 2: Implement the Supabase model store boundary

**Files:**
- Create: `modules/supabase_model_registry.py`
- Create: `tests/modules/test_supabase_model_registry.py`
- Modify: `requirements.txt`

**Step 1: Write failing store tests**

Define an SDK-independent API:

```python
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


class SupabaseModelStore:
    def current(self, key: ModelKey) -> RemoteGeneration | None: ...
    def download(self, generation: RemoteGeneration) -> bytes: ...
    def upload(self, generation_id: str, key: ModelKey, artifact: bytes) -> str: ...
    def activate(..., expected_generation_id: str | None) -> bool: ...
    def was_rejected(self, attempt: ModelAttempt) -> bool: ...
    def record_rejection(self, attempt: ModelAttempt, reason: str) -> None: ...
```

Using a fake Supabase SDK client, verify:

- table responses are validated rather than trusted;
- metadata scope must match the requested key;
- Storage download SHA-256 must match both row and metadata;
- upload uses immutable UUID paths, binary content type, and `upsert=false` so a reused UUID cannot overwrite an activated object;
- RPC receives the expected head and complete metadata;
- SDK exceptions are translated to `OSError` without including secret values;
- rejection lookup and upsert use the complete scope and fingerprint.

Run the targeted test and confirm RED because the module is absent.

**Step 2: Implement the minimal store**

- Import `supabase.create_client` lazily in `from_credentials()`.
- Accept an injected client in tests.
- Never retain or expose the raw secret outside the SDK client.
- Validate every response shape and integer revision.
- Use `ModelMetadata.from_dict()` and `ModelAttempt.to_dict()` as the serialization authority.
- Treat an empty head result as `None`; treat malformed or duplicate rows as an I/O contract error.

**Step 3: Run targeted GREEN tests**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_supabase_model_registry.py -q -k 'store'
```

**Step 4: Add the bounded dependency and commit**

Add `supabase>=2.18,<3` to `requirements.txt`, run the dependency declaration tests, then commit.

### Task 3: Add remote hydration and write-through promotion

**Files:**
- Modify: `modules/supabase_model_registry.py`
- Modify: `tests/modules/test_supabase_model_registry.py`
- Create: `tests/integration/test_supabase_model_lifecycle.py`

**Step 1: Write failing registry lifecycle tests**

Use an in-memory fake store and real sealed XGBoost candidates to verify:

1. No remote head returns `model_not_found`.
2. First acceptable candidate uploads and activates before `promoted=True` is returned.
3. A fresh registry instance downloads the remote generation and loads it without retraining.
4. Downloaded metadata, checksum, bundle scope, backend and runtime compatibility are revalidated by `ModelRegistry`.
5. A corrupt remote model affects only its key.
6. Upload/RPC failure returns a non-promoted decision and never exposes the candidate to prediction.
7. CAS conflict returns `remote_head_conflict` and preserves the remote winner.
8. A timeout after commit is reconciled by checking whether the submitted generation became head.
9. Deterministic rejection fingerprints survive a fresh registry instance.
10. Non-deterministic or malformed rejection reasons are not persisted.

Run the targeted tests and confirm RED because `SupabaseModelRegistry` is missing.

**Step 2: Implement `SupabaseModelRegistry`**

- Subclass `ModelRegistry` so all existing candidate admission and artifact validation remains authoritative.
- Use `TemporaryDirectory` for default local cache; allow an injected cache path in tests.
- Before load/promotion, hydrate the exact current remote generation into the local registry.
- Deserialize remote bytes only after SHA-256 and metadata validation.
- Install hydrated bundles through the existing atomic generation publisher, then call the existing `load()` compatibility checks.
- Cache generation IDs only for the lifetime of the registry.
- On local promotion success, read the validated active artifact, upload it, and invoke CAS activation.
- Return success only after remote activation or post-timeout reconciliation.
- Persist deterministic local rejection decisions to the remote rejection table.

**Step 3: Run module and integration GREEN tests**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest \
  tests/modules/test_supabase_model_registry.py \
  tests/integration/test_supabase_model_lifecycle.py -q
```

**Step 4: Commit**

```bash
git add modules/supabase_model_registry.py \
  tests/modules/test_supabase_model_registry.py \
  tests/integration/test_supabase_model_lifecycle.py
git commit -m "feat: persist XGBoost generations in Supabase"
```

### Task 4: Wire runtime configuration without changing the UI

**Files:**
- Modify: `config/config.py`
- Create: `modules/model_registry_factory.py`
- Modify: `dispatch_app_st.py`
- Modify: `tests/config/test_config.py`
- Create: `tests/modules/test_model_registry_factory.py`
- Modify: `tests/utils/test_audit_log.py`

**Step 1: Write failing configuration and factory tests**

Require:

- URL and secret both absent -> current local `ModelRegistry`;
- both present -> `SupabaseModelRegistry` with bucket default `dlr-models`;
- only one present -> explicit `ValueError`/configuration error;
- URL must be HTTPS and have no query, fragment, credentials, or unexpected project host;
- secret values never appear in `repr`, exception text, audit events, or Streamlit output;
- the page constructs the registry through the factory and retains the existing audit logger.

**Step 2: Add environment-backed config and factory**

Use root-level Streamlit secrets/environment variables:

```text
DLR_SUPABASE_URL
DLR_SUPABASE_SECRET_KEY
DLR_SUPABASE_MODEL_BUCKET
```

Keep local fallback for development only. Do not add a sidebar control or status label.

**Step 3: Run GREEN tests**

Run the new config/factory tests plus `tests/utils/test_audit_log.py`.

**Step 4: Commit**

```bash
git add config/config.py modules/model_registry_factory.py dispatch_app_st.py \
  tests/config/test_config.py tests/modules/test_model_registry_factory.py \
  tests/utils/test_audit_log.py
git commit -m "feat: select Supabase registry from server secrets"
```

### Task 5: Provision and verify the real Supabase project

**External resources:**
- Supabase project `ciapxhuldarsupmvrgwu`
- Private bucket `dlr-models`
- Three public-schema registry tables and activation RPC

**Step 1: Apply the committed migration in Supabase SQL Editor**

Use the exact committed SQL. Do not edit unrelated schemas or existing resources.

**Step 2: Verify project security state**

- bucket exists and is private;
- all three tables exist with RLS enabled;
- no `anon`/`authenticated` table or Storage policies grant access;
- RPC execution is limited to `service_role`;
- project security advisor shows no new issue caused by the migration.

**Step 3: Obtain the server-side secret and configure Streamlit Cloud**

Put URL, server-side secret, and bucket name in the deployed app's Secrets editor. Never place them in repository files.

**Step 4: Run a real round-trip smoke test**

Using the production store client and a uniquely scoped disposable test key:

- upload one valid small artifact;
- activate it;
- read the head and download it;
- verify the SHA-256;
- remove only the disposable smoke-test row/object after verification.

Do not remove or overwrite any user model.

### Task 6: Full verification, review, and deployment

**Files:**
- Modify: `README.md`
- Modify only as review findings require.

**Step 1: Document operations**

Document the three secret names, private bucket/table ownership, local-development fallback, and the physical-DLR behavior during Supabase outages. Do not include actual credentials.

**Step 2: Run the full quality gate**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest -q
/opt/homebrew/opt/python@3.11/bin/python3.11 -m ruff check --target-version py311 .
/opt/homebrew/opt/python@3.11/bin/python3.11 -m compileall -q config modules pages utils dispatch_app_st.py
git diff --check
```

**Step 3: Perform spec and code-quality reviews**

Review remote authority, checksum validation, secret handling, CAS semantics, per-key failure isolation, rejection persistence, tests, and migration security. Fix every finding and rerun the full gate.

**Step 4: Commit final documentation/fixes and push**

Push the feature branch to remote `main` only after all checks pass.

**Step 5: Verify the deployed app**

- main DLR page opens without exception;
- sag page remains independent and opens;
- a controlled training run persists a model to Supabase;
- a fresh app process loads that model without truth upload/retraining;
- Supabase generation/head rows and private Storage object agree on checksum.

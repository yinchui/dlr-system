import re
from pathlib import Path


MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "supabase"
    / "migrations"
    / "20260728000000_dlr_model_registry.sql"
)
SCOPE_COLUMNS = ("project_id", "line_id", "tower_id", "target")
TABLES = (
    "dlr_model_generations",
    "dlr_model_heads",
    "dlr_model_rejections",
)


def _migration_sql() -> str:
    assert MIGRATION_PATH.is_file(), f"missing migration: {MIGRATION_PATH}"
    return MIGRATION_PATH.read_text(encoding="utf-8").lower()


def _normalized_sql() -> str:
    return " ".join(_migration_sql().split())


def _table_definition(sql: str, table: str) -> str:
    start = sql.index(f"create table if not exists public.{table}")
    later_starts = [
        sql.find(f"create table if not exists public.{other}", start + 1)
        for other in TABLES
        if other != table
    ]
    end_candidates = [position for position in later_starts if position >= 0]
    end_candidates.extend(
        position
        for position in (
            sql.find("alter table", start + 1),
            sql.find("create or replace function", start + 1),
        )
        if position >= 0
    )
    end = min(end_candidates, default=len(sql))
    return sql[start:end]


def _activation_function(sql: str) -> str:
    start = sql.index(
        "create or replace function public.activate_dlr_model_generation"
    )
    end = sql.index("revoke", start)
    return sql[start:end]


def test_migration_idempotently_creates_private_binary_model_bucket():
    sql = _migration_sql()

    assert re.search(
        r"insert\s+into\s+storage\.buckets\s*\(\s*id\s*,\s*name\s*,\s*"
        r"public\s*,\s*allowed_mime_types\s*\)\s*values\s*\(\s*"
        r"'dlr-models'\s*,\s*'dlr-models'\s*,\s*false\s*,\s*"
        r"array\s*\[\s*'application/octet-stream'\s*\]\s*\)",
        sql,
    )
    assert re.search(r"on\s+conflict\s*\(\s*id\s*\)\s+do\s+update", sql)
    assert "public = false" in sql
    assert "allowed_mime_types = array['application/octet-stream']" in sql


def test_migration_creates_registry_tables_with_scope_and_constraints():
    sql = _migration_sql()

    for table in TABLES:
        definition = _table_definition(sql, table)
        for column in SCOPE_COLUMNS:
            assert re.search(rf"\b{column}\s+text\s+not\s+null\b", definition)

    generations = _table_definition(sql, "dlr_model_generations")
    heads = _table_definition(sql, "dlr_model_heads")
    rejections = _table_definition(sql, "dlr_model_rejections")

    assert "primary key (project_id, line_id, tower_id, target)" in heads
    assert "unique (project_id, line_id, tower_id, target, attempt_fingerprint)" in (
        rejections
    )
    assert re.search(
        r"check\s*\(\s*target\s+in\s*\(\s*'wind_speed'\s*,\s*"
        r"'ambient_temp'\s*\)\s*\)",
        generations,
    )
    assert re.search(
        r"check\s*\(\s*status\s+in\s*\(\s*'active_provisional'\s*,\s*"
        r"'active'\s*\)\s*\)",
        generations,
    )
    assert re.search(
        r"check\s*\(\s*model_checksum\s*~\s*'\^\[0-9a-f\]\{64\}\$'\s*\)",
        generations,
    )


def test_jsonb_integrity_constraints_are_replayable_and_fail_closed():
    sql = _normalized_sql()

    assert (
        "alter table public.dlr_model_generations drop constraint if exists "
        "dlr_model_generations_metadata_check;"
    ) in sql
    assert (
        "alter table public.dlr_model_rejections drop constraint if exists "
        "dlr_model_rejections_attempt_check;"
    ) in sql
    assert re.search(
        r"add constraint dlr_model_generations_metadata_check check \(\(.*?"
        r"metadata \?& array\[.*?'project_id'.*?'line_id'.*?'tower_id'.*?"
        r"'target'.*?'model_version'.*?'checksum'.*?'status'.*?\].*?"
        r"\) is true\);",
        sql,
    )
    assert re.search(
        r"add constraint dlr_model_rejections_attempt_check check \(\(.*?"
        r"attempt \?& array\[.*?'project_id'.*?'line_id'.*?'tower_id'.*?"
        r"'target'.*?\].*?\) is true\);",
        sql,
    )


def test_migration_enables_rls_without_client_policies():
    sql = _normalized_sql()

    for table in TABLES:
        assert f"alter table public.{table} enable row level security;" in sql

    assert "create policy" not in sql
    assert "alter policy" not in sql
    assert not re.search(r"\bto\s+(anon|authenticated)\b", sql)


def test_activation_rpc_has_fixed_signature_and_security_boundary():
    sql = _normalized_sql()
    function = _activation_function(sql)

    expected_signature = """create or replace function public.activate_dlr_model_generation(
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
    ) returns boolean"""
    assert " ".join(expected_signature.split()) in sql
    assert "security definer" in function
    assert "set search_path = ''" in function

    function_identity = (
        r"public\.activate_dlr_model_generation\s*\(\s*uuid\s*,\s*text\s*,\s*"
        r"text\s*,\s*text\s*,\s*text\s*,\s*text\s*,\s*text\s*,\s*text\s*,\s*"
        r"jsonb\s*,\s*text\s*,\s*uuid\s*\)"
    )
    assert re.search(
        rf"revoke\s+all\s+on\s+function\s+{function_identity}"
        r"\s+from\s+public\s*;",
        sql,
    )
    for role in ("anon", "authenticated"):
        assert re.search(
            rf"revoke\s+all\s+on\s+function\s+{function_identity}"
            rf"\s+from\s+{role}\s*;",
            sql,
        )
    assert re.search(
        rf"grant\s+execute\s+on\s+function\s+{function_identity}"
        r"\s+to\s+service_role\s*;",
        sql,
    )
    assert not re.search(
        rf"grant\s+execute\s+on\s+function\s+{function_identity}"
        r"\s+to\s+(anon|authenticated)\b",
        sql,
    )


def test_activation_rpc_locks_and_checks_expected_head_before_publication():
    sql = _normalized_sql()
    function = " ".join(_activation_function(sql).split())

    advisory_lock = function.index("pg_catalog.pg_advisory_xact_lock")
    head_lock = function.index("from public.dlr_model_heads as head")
    for_update = function.index("for update", head_lock)
    cas_check = function.index(
        "v_current_generation_id is distinct from p_expected_generation_id"
    )
    false_result = function.index("return false", cas_check)
    generation_insert = function.index(
        "insert into public.dlr_model_generations", false_result
    )
    head_upsert = function.index("insert into public.dlr_model_heads", generation_insert)

    assert advisory_lock < head_lock < for_update < cas_check < false_result
    assert false_result < generation_insert < head_upsert
    assert "on conflict (project_id, line_id, tower_id, target) do update" in (
        function
    )
    assert "revision = public.dlr_model_heads.revision + 1" in function
    assert function.rindex("return true") > head_upsert

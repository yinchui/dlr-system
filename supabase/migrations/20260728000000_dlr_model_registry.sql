insert into storage.buckets (id, name, public, allowed_mime_types)
values (
    'dlr-models',
    'dlr-models',
    false,
    array['application/octet-stream']
)
on conflict (id) do update
set
    name = excluded.name,
    public = false,
    allowed_mime_types = array['application/octet-stream'];

create table if not exists public.dlr_model_generations (
    generation_id uuid primary key,
    project_id text not null,
    line_id text not null,
    tower_id text not null,
    target text not null,
    model_version text not null,
    storage_path text not null,
    model_checksum text not null,
    metadata jsonb not null,
    status text not null,
    created_at timestamptz not null default now(),
    constraint dlr_model_generations_scope_generation_key
        unique (generation_id, project_id, line_id, tower_id, target),
    constraint dlr_model_generations_storage_path_key unique (storage_path),
    constraint dlr_model_generations_target_check
        check (target in ('wind_speed', 'ambient_temp')),
    constraint dlr_model_generations_status_check
        check (status in ('active_provisional', 'active')),
    constraint dlr_model_generations_checksum_check
        check (model_checksum ~ '^[0-9a-f]{64}$'),
    constraint dlr_model_generations_metadata_check
        check (
            jsonb_typeof(metadata) = 'object'
            and metadata ->> 'project_id' = project_id
            and metadata ->> 'line_id' = line_id
            and metadata ->> 'tower_id' = tower_id
            and metadata ->> 'target' = target
            and metadata ->> 'model_version' = model_version
            and metadata ->> 'checksum' = model_checksum
            and metadata ->> 'status' = status
        )
);

create table if not exists public.dlr_model_heads (
    project_id text not null,
    line_id text not null,
    tower_id text not null,
    target text not null,
    generation_id uuid not null,
    revision bigint not null default 1,
    updated_at timestamptz not null default now(),
    constraint dlr_model_heads_pkey
        primary key (project_id, line_id, tower_id, target),
    constraint dlr_model_heads_target_check
        check (target in ('wind_speed', 'ambient_temp')),
    constraint dlr_model_heads_revision_check check (revision > 0),
    constraint dlr_model_heads_generation_scope_fkey
        foreign key (generation_id, project_id, line_id, tower_id, target)
        references public.dlr_model_generations (
            generation_id,
            project_id,
            line_id,
            tower_id,
            target
        )
);

create table if not exists public.dlr_model_rejections (
    rejection_id bigint generated always as identity primary key,
    project_id text not null,
    line_id text not null,
    tower_id text not null,
    target text not null,
    attempt_fingerprint text not null,
    champion_context_hash text,
    reason text not null,
    attempt jsonb not null,
    created_at timestamptz not null default now(),
    constraint dlr_model_rejections_scope_attempt_key
        unique (project_id, line_id, tower_id, target, attempt_fingerprint),
    constraint dlr_model_rejections_target_check
        check (target in ('wind_speed', 'ambient_temp')),
    constraint dlr_model_rejections_fingerprint_check
        check (attempt_fingerprint ~ '^[0-9a-f]{64}$'),
    constraint dlr_model_rejections_champion_context_check
        check (
            champion_context_hash is null
            or champion_context_hash ~ '^[0-9a-f]{64}$'
        ),
    constraint dlr_model_rejections_reason_check
        check (length(btrim(reason)) > 0),
    constraint dlr_model_rejections_attempt_check
        check (
            jsonb_typeof(attempt) = 'object'
            and attempt ->> 'project_id' = project_id
            and attempt ->> 'line_id' = line_id
            and attempt ->> 'tower_id' = tower_id
            and attempt ->> 'target' = target
        )
);

alter table public.dlr_model_generations enable row level security;
alter table public.dlr_model_heads enable row level security;
alter table public.dlr_model_rejections enable row level security;

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
as $function$
declare
    v_current_generation_id uuid;
begin
    perform pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            pg_catalog.jsonb_build_array(
                p_project_id,
                p_line_id,
                p_tower_id,
                p_target
            )::pg_catalog.text,
            0
        )
    );

    select head.generation_id
    into v_current_generation_id
    from public.dlr_model_heads as head
    where head.project_id = p_project_id
        and head.line_id = p_line_id
        and head.tower_id = p_tower_id
        and head.target = p_target
    for update;

    if v_current_generation_id is distinct from p_expected_generation_id then
        return false;
    end if;

    insert into public.dlr_model_generations (
        generation_id,
        project_id,
        line_id,
        tower_id,
        target,
        model_version,
        storage_path,
        model_checksum,
        metadata,
        status
    ) values (
        p_generation_id,
        p_project_id,
        p_line_id,
        p_tower_id,
        p_target,
        p_model_version,
        p_storage_path,
        p_model_checksum,
        p_metadata,
        p_status
    );

    insert into public.dlr_model_heads (
        project_id,
        line_id,
        tower_id,
        target,
        generation_id,
        revision,
        updated_at
    ) values (
        p_project_id,
        p_line_id,
        p_tower_id,
        p_target,
        p_generation_id,
        1,
        pg_catalog.now()
    )
    on conflict (project_id, line_id, tower_id, target) do update
    set
        generation_id = excluded.generation_id,
        revision = public.dlr_model_heads.revision + 1,
        updated_at = pg_catalog.now();

    return true;
end;
$function$;

revoke all on function public.activate_dlr_model_generation(
    uuid,
    text,
    text,
    text,
    text,
    text,
    text,
    text,
    jsonb,
    text,
    uuid
) from public;

grant execute on function public.activate_dlr_model_generation(
    uuid,
    text,
    text,
    text,
    text,
    text,
    text,
    text,
    jsonb,
    text,
    uuid
) to service_role;

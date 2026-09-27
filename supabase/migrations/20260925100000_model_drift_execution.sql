begin;

create table public.model_drift_execution (
    id bigint generated always as identity primary key,
    execution_id uuid not null unique default gen_random_uuid(),
    started_at timestamptz not null,
    completed_at timestamptz,
    status text not null default 'success',
    model_id bigint references public.model(id),
    model_version text,
    reference_window jsonb not null default '{}'::jsonb,
    current_window jsonb not null default '{}'::jsonb,
    analyzed_features jsonb not null default '[]'::jsonb,
    feature_psi jsonb not null default '{}'::jsonb,
    drifted_features jsonb not null default '[]'::jsonb,
    thresholds jsonb not null default '{}'::jsonb,
    performance jsonb not null default '{}'::jsonb,
    drift_score numeric,
    drift_detected boolean,
    result jsonb not null default '{}'::jsonb,
    error_message text,
    check (status in ('success', 'error')),
    check (completed_at is null or completed_at >= started_at)
);

create index model_drift_execution_started_at_idx
    on public.model_drift_execution (started_at desc);

create index model_drift_execution_status_idx
    on public.model_drift_execution (status, started_at desc);

alter table public.model_drift_execution enable row level security;

commit;

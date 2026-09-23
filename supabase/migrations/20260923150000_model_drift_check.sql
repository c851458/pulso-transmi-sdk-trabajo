begin;

create table public.model_drift_check (
    id bigint generated always as identity primary key,
    checked_at timestamptz not null default now(),
    model_id bigint references public.model(id),
    model_version text,
    reference_window jsonb not null,
    current_window jsonb not null,
    thresholds jsonb not null,
    feature_psi jsonb not null,
    drifted_features jsonb not null default '[]'::jsonb,
    performance jsonb not null default '{}'::jsonb,
    drift_alert boolean not null default false,
    retrain boolean not null default false,
    reason text not null
);

create index model_drift_check_checked_at_idx
    on public.model_drift_check (checked_at desc);

alter table public.model_drift_check enable row level security;

commit;

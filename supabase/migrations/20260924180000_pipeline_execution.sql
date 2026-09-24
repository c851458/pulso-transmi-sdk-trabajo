begin;

create table public.pipeline_execution (
    run_id text primary key,
    model_id bigint not null references public.model(id),
    cycle_id text not null,
    model_version text not null,
    generated_at timestamptz not null,
    metrics jsonb not null,
    prediction_count integer not null,
    payload_sha256 text not null,
    status text not null,
    api_submission_id text,
    api_response jsonb,
    error text,
    confirmed_at timestamptz,
    check (prediction_count > 0),
    check (status in ('prepared', 'confirmed', 'partial_failure', 'failed'))
);

create index pipeline_execution_cycle_idx
    on public.pipeline_execution (cycle_id, generated_at desc);

alter table public.pipeline_execution enable row level security;

commit;

begin;

create table public.pipeline_sync_status (
    id text primary key,
    status text not null,
    started_at timestamptz,
    last_updated_at timestamptz,
    records_received integer not null default 0,
    records_inserted integer not null default 0,
    records_updated integer not null default 0,
    records_unchanged integer not null default 0,
    error text,
    updated_at timestamptz not null default now(),
    check (status in ('SYNCING', 'SUCCESS', 'FAILED', 'STALE', 'WAITING_FOR_OPEN_CYCLE')),
    check (records_received >= 0),
    check (records_inserted >= 0),
    check (records_updated >= 0),
    check (records_unchanged >= 0)
);

alter table public.pipeline_sync_status enable row level security;

commit;

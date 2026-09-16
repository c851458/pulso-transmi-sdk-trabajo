begin;

create table public.data_source (
    id bigint generated always as identity primary key,
    name text not null,
    source_type text not null,
    base_url text not null,
    created_at timestamptz not null default now()
);

create table public.data_cut (
    id bigint generated always as identity primary key,
    source_id bigint not null references public.data_source(id),
    dataset_version text not null,
    queried_at timestamptz not null default now(),
    period_start timestamptz not null,
    period_end timestamptz not null,
    final_cursor text,
    metadata_sha256 text,
    code_commit text,
    check (period_end >= period_start),
    unique (source_id, dataset_version, period_start, period_end)
);

create table public.station (
    station_id text primary key,
    station_name text not null,
    corridor text not null,
    latitude numeric(9, 6) not null,
    longitude numeric(9, 6) not null,
    check (latitude between -90 and 90),
    check (longitude between -180 and 180)
);

create table public.demand_observation (
    id bigint generated always as identity primary key,
    data_cut_id bigint not null references public.data_cut(id),
    station_id text not null references public.station(station_id),
    observed_at timestamptz not null,
    demand numeric not null,
    check (demand >= 0),
    unique (data_cut_id, station_id, observed_at)
);

create index demand_observation_time_idx
    on public.demand_observation (station_id, observed_at);

create table public.context_observation (
    id bigint generated always as identity primary key,
    data_cut_id bigint not null references public.data_cut(id),
    observed_at timestamptz not null,
    event_intensity numeric not null,
    rain_forecast numeric not null,
    rain_mm numeric not null,
    temperature_c numeric not null,
    temperature_forecast numeric not null,
    check (event_intensity between 0 and 1),
    check (rain_forecast >= 0),
    check (rain_mm >= 0),
    unique (data_cut_id, observed_at)
);

create index context_observation_time_idx
    on public.context_observation (observed_at);

create table public.raw_ingestion (
    id bigint generated always as identity primary key,
    data_cut_id bigint not null references public.data_cut(id),
    endpoint text not null,
    received_at timestamptz not null default now(),
    payload jsonb not null,
    payload_sha256 text not null
);

create index raw_ingestion_cut_endpoint_idx
    on public.raw_ingestion (data_cut_id, endpoint);

create table public.feature_snapshot (
    id bigint generated always as identity primary key,
    data_cut_id bigint not null references public.data_cut(id),
    station_id text not null references public.station(station_id),
    target_at timestamptz not null,
    feature_version text not null,
    feature_values jsonb not null,
    generated_at timestamptz not null default now(),
    unique (data_cut_id, station_id, target_at, feature_version)
);

create table public.model (
    id bigint generated always as identity primary key,
    version text not null unique,
    algorithm text not null,
    status text not null,
    code_commit text,
    trained_at timestamptz not null,
    check (status in ('candidate', 'active', 'retired', 'failed'))
);

create table public.training_run (
    id bigint generated always as identity primary key,
    model_id bigint not null references public.model(id),
    data_cut_id bigint not null references public.data_cut(id),
    cutoff_at timestamptz not null,
    parameters jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,
    started_at timestamptz not null,
    finished_at timestamptz,
    check (finished_at is null or finished_at >= started_at)
);

create table public.prediction (
    id bigint generated always as identity primary key,
    model_id bigint not null references public.model(id),
    station_id text not null references public.station(station_id),
    target_at timestamptz not null,
    generated_at timestamptz not null,
    horizon_periods integer not null,
    prediction numeric not null,
    actual_value numeric,
    check (horizon_periods > 0),
    check (prediction >= 0),
    unique (model_id, station_id, target_at, horizon_periods)
);

create index prediction_station_time_idx
    on public.prediction (station_id, target_at);

create table public.monitoring_metric (
    id bigint generated always as identity primary key,
    prediction_id bigint not null references public.prediction(id),
    measured_at timestamptz not null default now(),
    wape numeric,
    accuracy numeric,
    mae numeric,
    rmse numeric,
    bias numeric,
    drift_score numeric,
    alert boolean not null default false,
    check (wape is null or wape >= 0),
    check (mae is null or mae >= 0),
    check (rmse is null or rmse >= 0)
);

-- Supabase exposes public tables through its API. Keep access closed until
-- explicit policies are added for the intended application roles.
alter table public.data_source enable row level security;
alter table public.data_cut enable row level security;
alter table public.station enable row level security;
alter table public.demand_observation enable row level security;
alter table public.context_observation enable row level security;
alter table public.raw_ingestion enable row level security;
alter table public.feature_snapshot enable row level security;
alter table public.model enable row level security;
alter table public.training_run enable row level security;
alter table public.prediction enable row level security;
alter table public.monitoring_metric enable row level security;

commit;

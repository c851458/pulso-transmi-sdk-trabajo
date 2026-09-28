begin;

-- Support the incremental ingestion and drift keyset scans without changing
-- rows, constraints, RLS, or the application decision logic.
create index if not exists demand_observation_cut_time_id_idx
    on public.demand_observation (data_cut_id, observed_at, id);

create index if not exists context_observation_cut_time_id_idx
    on public.context_observation (data_cut_id, observed_at, id);

commit;

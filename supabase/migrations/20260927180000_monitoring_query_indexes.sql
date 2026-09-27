begin;

-- Non-destructive indexes for the existing monitoring access patterns.
-- These do not alter rows, constraints, RLS, or application decision logic.
create index if not exists demand_observation_observed_at_idx
    on public.demand_observation (observed_at desc);

create index if not exists model_trained_at_idx
    on public.model (trained_at desc);

create index if not exists training_run_model_started_at_idx
    on public.training_run (model_id, started_at desc);

create index if not exists prediction_model_target_at_idx
    on public.prediction (model_id, target_at desc)
    where actual_value is not null;

create index if not exists pipeline_execution_generated_at_idx
    on public.pipeline_execution (generated_at desc);

commit;

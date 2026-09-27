begin;

alter table public.model
    add column mlflow_run_id text,
    add column mlflow_model_name text,
    add column mlflow_model_version text;

alter table public.training_run
    add column mlflow_run_id text;

create index model_mlflow_run_idx
    on public.model (mlflow_run_id);

create index training_run_mlflow_run_idx
    on public.training_run (mlflow_run_id);

commit;

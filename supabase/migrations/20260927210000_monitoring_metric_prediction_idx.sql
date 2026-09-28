begin;

-- Deleting a prediction checks the monitoring_metric foreign key. Without this
-- index each check scans the whole table, so prune_model_history exceeded the
-- 8 s statement_timeout of the REST API.
create index if not exists monitoring_metric_prediction_id_idx
    on public.monitoring_metric (prediction_id);

commit;

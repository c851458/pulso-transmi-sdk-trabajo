begin;

-- Every retraining stores a compressed artifact plus one evaluation prediction
-- and one monitoring_metric row per validation sample. Keeping that history for
-- every superseded model filled the 2 GB free-tier disk. Only the current model
-- needs its evaluation rows (drift reads them) and its artifact (publish-only
-- restores it); older artifacts remain available in MLflow.
--
-- Forecast predictions submitted to the API (actual_value is null), training
-- runs, executions and observations are never removed.
create or replace function public.prune_model_history(keep_model_id bigint)
returns table (metrics_deleted bigint, predictions_deleted bigint, models_retired bigint)
language plpgsql
security definer
set search_path = public
as $$
begin
    if not exists (select 1 from public.model where id = keep_model_id) then
        raise exception 'model % does not exist', keep_model_id;
    end if;

    delete from public.monitoring_metric mm
    using public.prediction p
    where mm.prediction_id = p.id
      and p.model_id <> keep_model_id
      and p.actual_value is not null;
    get diagnostics metrics_deleted = row_count;

    delete from public.prediction p
    where p.model_id <> keep_model_id
      and p.actual_value is not null
      and not exists (select 1 from public.monitoring_metric mm where mm.prediction_id = p.id);
    get diagnostics predictions_deleted = row_count;

    update public.model
    set status = 'retired', artifact_base64 = null
    where id <> keep_model_id
      and (status = 'active' or artifact_base64 is not null);
    get diagnostics models_retired = row_count;

    return next;
end;
$$;

revoke all on function public.prune_model_history(bigint) from public, anon, authenticated;
grant execute on function public.prune_model_history(bigint) to service_role;

commit;

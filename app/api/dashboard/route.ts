import { NextResponse } from "next/server";

const TABLES = ["model", "training_run", "model_drift_check", "model_drift_execution", "pipeline_execution", "pipeline_sync_status", "training_run (timeline)", "model_drift_check (timeline)"] as const;
const TIMELINE_DAYS = 7;
type Row = Record<string, any>;
export const revalidate = 300;

function number(value: unknown): number | null {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

async function table(name: string, query: string) {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) throw new Error("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY en el entorno del servidor");
  const response = await fetch(`${url.replace(/\/$/, "")}/rest/v1/${name}?${query}`, {
    headers: { apikey: key, Authorization: `Bearer ${key}` },
    signal: AbortSignal.timeout(10000),
    next: { revalidate: 300 },
  });
  if (!response.ok) throw new Error(`Supabase ${name}: ${response.status}`);
  return response.json() as Promise<Row[]>;
}

function metric(row: Row | undefined, key: string) {
  return number(row?.metrics?.[key] ?? row?.metrics?.test?.[key] ?? row?.performance?.[key]);
}

export async function GET() {
  try {
    const since = new Date(Date.now() - TIMELINE_DAYS * 86400000).toISOString();
    const results = await Promise.allSettled([
      table("model", "select=id,version,algorithm,status,trained_at,mlflow_run_id,mlflow_model_name,mlflow_model_version&order=trained_at.desc&limit=20"),
      table("training_run", "select=id,model_id,started_at,finished_at,metrics&order=started_at.desc&limit=50"),
      table("model_drift_check", "select=id,checked_at,model_id,model_version,reference_window,current_window,thresholds,feature_psi,drifted_features,performance,drift_alert,retrain,reason&order=checked_at.desc&limit=50"),
      table("model_drift_execution", "select=id,execution_id,started_at,completed_at,status,model_id,model_version,reference_window,current_window,analyzed_features,feature_psi,drifted_features,thresholds,performance,drift_score,drift_detected,error_message&order=started_at.desc&limit=20"),
      table("pipeline_execution", "select=run_id,model_id,cycle_id,model_version,generated_at,metrics,prediction_count,payload_sha256,status,error,confirmed_at&order=generated_at.desc&limit=20"),
      table("pipeline_sync_status", "select=id,status,started_at,last_updated_at,records_received,records_inserted,records_updated,records_unchanged,error,updated_at&order=updated_at.desc&limit=1"),
      // Only the test metrics the timeline plots, not the full metrics JSONB of every run.
      table("training_run", `select=model_id,started_at,wape:metrics->test->wape,mae:metrics->test->mae,rmse:metrics->test->rmse,r2:metrics->test->r2,accuracy:metrics->test->accuracy&started_at=gte.${since}&order=started_at.asc&limit=1500`),
      table("model_drift_check", `select=checked_at,model_version,feature_psi,drifted_features,thresholds,performance,drift_alert,retrain&checked_at=gte.${since}&order=checked_at.asc&limit=1500`),
    ]);
    const read = (index: number) => {
      const result = results[index];
      return result.status === "fulfilled" ? result.value : [];
    };
    const errors = results.flatMap((result, index) => result.status === "rejected" ? [{ table: TABLES[index], message: result.reason instanceof Error ? result.reason.message : "No fue posible leer la tabla" }] : []);
    const [models, training, checks, executions, pipeline, sync, trainingTimeline, checkTimeline] = [0, 1, 2, 3, 4, 5, 6, 7].map(read);

    const active = models.find((row) => row.status === "active") ?? models[0] ?? null;
    const activeTraining = training.find((row) => row.model_id === active?.id) ?? training[0] ?? null;
    const previousModel = models.find((row) => row.id !== active?.id && (!active || row.trained_at <= active.trained_at)) ?? null;
    const previousTraining = previousModel ? training.find((row) => row.model_id === previousModel.id) ?? null : null;
    const latestCheck = checks[0] ?? null;
    const latestExecution = executions[0] ?? null;
    const latestSuccessful = executions.find((row) => row.status === "success") ?? null;
    const latestFailed = executions.find((row) => row.status === "error") ?? null;
    // The check is the authoritative decision record; execution stores the audit trail.
    const source = latestCheck ?? latestExecution;
    const performance = source?.performance ?? latestCheck?.performance ?? {};
    const featurePsi = source?.feature_psi ?? latestCheck?.feature_psi ?? {};
    const drifted = Array.isArray(source?.drifted_features) ? source.drifted_features : [];
    const threshold = number(source?.thresholds?.psi ?? latestCheck?.thresholds?.psi) ?? 0.2;
    const ratioThreshold = number(source?.thresholds?.performance_ratio ?? latestCheck?.thresholds?.performance_ratio) ?? 1.25;
    const required = number(source?.thresholds?.drifted_features_required ?? latestCheck?.thresholds?.drifted_features_required) ?? 2;

    const history = executions.map((row) => {
      const p = row.performance ?? {};
      const psi = row.feature_psi ?? {};
      return {
        id: row.execution_id ?? String(row.id), date: row.started_at, model: row.model_id, version: row.model_version ?? "—",
        psi: number(row.drift_score ?? Math.max(...Object.values(psi).map((v) => number(v) ?? 0), 0)), drifted: Array.isArray(row.drifted_features) ? row.drifted_features.length : 0,
        wape: number(p.recent_wape), baseline: number(p.baseline_wape), ratio: number(p.ratio), alert: Boolean(row.drift_detected), retrain: Boolean(row.result?.retrain), status: row.status,
      };
    });

    return NextResponse.json({
      updatedAt: new Date().toISOString(), sourceErrors: errors,
      model: active ? { id: active.id, version: active.version, status: active.status, algorithm: active.algorithm, trainedAt: active.trained_at, mlflowRunId: active.mlflow_run_id, mlflowModelName: active.mlflow_model_name, mlflowModelVersion: active.mlflow_model_version } : null,
      training: activeTraining ? { metrics: activeTraining.metrics ?? {}, startedAt: activeTraining.started_at, finishedAt: activeTraining.finished_at } : null,
      previousTraining: previousTraining ? { metrics: previousTraining.metrics ?? {}, version: previousModel?.version, trainedAt: previousModel?.trained_at } : null,
      timeline: {
        days: TIMELINE_DAYS,
        retrains: trainingTimeline.map((row) => ({ date: row.started_at, model: row.model_id, wape: number(row.wape), mae: number(row.mae), rmse: number(row.rmse), r2: number(row.r2), accuracy: number(row.accuracy) })),
        checks: checkTimeline.map((row) => ({
          date: row.checked_at, version: row.model_version ?? null,
          maxPsi: Math.max(...Object.values(row.feature_psi ?? {}).map((v) => number(v) ?? 0), 0),
          drifted: Array.isArray(row.drifted_features) ? row.drifted_features : [],
          required: number(row.thresholds?.drifted_features_required),
          recentWape: number(row.performance?.recent_wape), accuracy: number(row.performance?.accuracy), baselineWape: number(row.performance?.baseline_wape), ratio: number(row.performance?.ratio),
          alert: Boolean(row.drift_alert), retrain: Boolean(row.retrain),
        })),
      },
      drift: { checkedAt: source?.checked_at ?? source?.started_at ?? null, psi: featurePsi, drifted, maxPsi: number(source?.drift_score ?? Math.max(...Object.values(featurePsi).map((v) => number(v) ?? 0), 0)), alert: Boolean(source?.drift_alert ?? source?.drift_detected), retrain: Boolean(source?.retrain ?? source?.result?.retrain), reason: source?.reason ?? source?.result?.reason ?? "Sin explicación registrada", reference: source?.reference_window ?? {}, current: source?.current_window ?? {}, thresholds: { psi: threshold, ratio: ratioThreshold, required } },
      performance: { accuracy: number(performance.accuracy), accuracySource: performance.accuracy_source ?? null, accuracyThreshold: number(source?.thresholds?.accuracy ?? latestCheck?.thresholds?.accuracy) ?? 79, current: number(performance.recent_wape), baseline: number(performance.baseline_wape) ?? metric(activeTraining, "wape"), ratio: number(performance.ratio), gate: performance.gate ?? (performance.alert == null ? "unavailable" : performance.alert ? "fail" : "pass"), history: checks.slice().reverse().map((row) => ({ date: row.checked_at, wape: number(row.performance?.recent_wape), baseline: number(row.performance?.baseline_wape), ratio: number(row.performance?.ratio) })) },
      executions: { latest: latestExecution ? { date: latestExecution.started_at, status: latestExecution.status } : null, lastSuccess: latestSuccessful?.completed_at ?? null, lastFailure: latestFailed?.completed_at ?? null, failedCount: executions.filter((row) => row.status === "error").length, history },
      pipeline: { latest: pipeline[0] ?? null, sync: sync[0] ?? null },
      models: models.map((row) => ({ id: row.id, version: row.version, status: row.status, algorithm: row.algorithm, trainedAt: row.trained_at, metrics: training.find((run) => run.model_id === row.id)?.metrics ?? {}, mlflowRunId: row.mlflow_run_id, mlflowModelVersion: row.mlflow_model_version })),
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "No fue posible consultar Supabase" }, { status: 500 });
  }
}

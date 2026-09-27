import { NextResponse } from "next/server";

const TABLES = ["model", "training_run", "model_drift_check", "model_drift_execution", "pipeline_execution", "pipeline_sync_status"] as const;
type Row = Record<string, any>;

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
    next: { revalidate: 60 },
  });
  if (!response.ok) throw new Error(`Supabase ${name}: ${response.status}`);
  return response.json() as Promise<Row[]>;
}

function metric(row: Row | undefined, key: string) {
  return number(row?.metrics?.[key] ?? row?.performance?.[key]);
}

export async function GET() {
  try {
    const results = await Promise.allSettled([
      table("model", "select=*&order=trained_at.desc&limit=100"),
      table("training_run", "select=*&order=started_at.desc&limit=100"),
      table("model_drift_check", "select=*&order=checked_at.desc&limit=100"),
      table("model_drift_execution", "select=id,execution_id,started_at,completed_at,status,model_id,model_version,reference_window,current_window,analyzed_features,feature_psi,drifted_features,thresholds,performance,drift_score,drift_detected,error_message&order=started_at.desc&limit=50"),
      table("pipeline_execution", "select=*&order=generated_at.desc&limit=100"),
      table("pipeline_sync_status", "select=*&order=updated_at.desc&limit=10"),
    ]);
    const read = (index: number) => {
      const result = results[index];
      return result.status === "fulfilled" ? result.value : [];
    };
    const errors = results.flatMap((result, index) => result.status === "rejected" ? [{ table: TABLES[index], message: result.reason instanceof Error ? result.reason.message : "No fue posible leer la tabla" }] : []);
    const [models, training, checks, executions, pipeline, sync] = [0, 1, 2, 3, 4, 5].map(read);

    const active = models.find((row) => row.status === "active") ?? models[0] ?? null;
    const activeTraining = training.find((row) => row.model_id === active?.id) ?? training[0] ?? null;
    const latestCheck = checks[0] ?? null;
    const latestExecution = executions[0] ?? null;
    const latestSuccessful = executions.find((row) => row.status === "success") ?? null;
    const latestFailed = executions.find((row) => row.status === "error") ?? null;
    const source = latestExecution ?? latestCheck;
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
      drift: { checkedAt: source?.checked_at ?? source?.started_at ?? null, psi: featurePsi, drifted, maxPsi: number(source?.drift_score ?? Math.max(...Object.values(featurePsi).map((v) => number(v) ?? 0), 0)), alert: Boolean(source?.drift_alert ?? source?.drift_detected), retrain: Boolean(source?.retrain ?? source?.result?.retrain), reason: source?.reason ?? source?.result?.reason ?? "Sin explicación registrada", reference: source?.reference_window ?? {}, current: source?.current_window ?? {}, thresholds: { psi: threshold, ratio: ratioThreshold, required } },
      performance: { current: number(performance.recent_wape), baseline: number(performance.baseline_wape) ?? metric(activeTraining, "wape"), ratio: number(performance.ratio), history: checks.slice().reverse().map((row) => ({ date: row.checked_at, wape: number(row.performance?.recent_wape), baseline: number(row.performance?.baseline_wape), ratio: number(row.performance?.ratio) })) },
      executions: { latest: latestExecution ? { date: latestExecution.started_at, status: latestExecution.status } : null, lastSuccess: latestSuccessful?.completed_at ?? null, lastFailure: latestFailed?.completed_at ?? null, failedCount: executions.filter((row) => row.status === "error").length, history },
      pipeline: { latest: pipeline[0] ?? null, sync: sync[0] ?? null },
      models: models.map((row) => ({ id: row.id, version: row.version, status: row.status, algorithm: row.algorithm, trainedAt: row.trained_at, metrics: training.find((run) => run.model_id === row.id)?.metrics ?? {}, mlflowRunId: row.mlflow_run_id, mlflowModelVersion: row.mlflow_model_version })),
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "No fue posible consultar Supabase" }, { status: 500 });
  }
}

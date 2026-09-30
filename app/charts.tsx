"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export type Point = { t: number; y: number | null };
export type Series = { key: string; label: string; color: string; points: Point[]; step?: boolean; format: (value: number) => string };
export type Marker = { t: number; y: number; label: string };
export type RugEvent = { t: number; label: string; tone: "muted" | "red" };

const M = { top: 14, right: 14, bottom: 26, left: 46 };
const LABEL_SPACE = 104;

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function niceStep(raw: number) {
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((f) => f * magnitude).reduce((best, v) => Math.abs(v - raw) < Math.abs(best - raw) ? v : best);
}

function timeTicks([t0, t1]: [number, number], innerW: number) {
  const span = t1 - t0;
  const hour = 3600000;
  // Keep at least ~80px between labels so narrow screens do not overlap them.
  const steps = [4, 6, 12, 24, 48].map((h) => h * hour);
  const step = steps.find((candidate) => (candidate / span) * innerW >= 80) ?? steps.at(-1)!;
  const start = new Date(t0);
  start.setMinutes(0, 0, 0);
  if (step >= 24 * hour) start.setHours(0);
  const ticks: number[] = [];
  for (let t = start.getTime(); t <= t1; t += hour) {
    const d = new Date(t);
    const aligned = step >= 24 * hour ? d.getHours() === 0 && Math.round((t - start.getTime()) / 86400000) % (step / (24 * hour)) === 0 : d.getHours() % (step / hour) === 0;
    if (t >= t0 && aligned) ticks.push(t);
  }
  const format = new Intl.DateTimeFormat("es-CO", step >= 24 * hour ? { day: "numeric", month: "short" } : step >= 12 * hour ? { day: "numeric", month: "short", hour: "2-digit" } : { hour: "2-digit", minute: "2-digit" });
  return ticks.map((t) => ({ t, label: format.format(new Date(t)) }));
}

const tooltipDate = new Intl.DateTimeFormat("es-CO", { dateStyle: "medium", timeStyle: "short" });

function valueAt(series: Series, t: number) {
  const points = series.points.filter((p) => p.y != null);
  if (!points.length) return null;
  if (series.step) {
    const previous = points.filter((p) => p.t <= t).at(-1);
    return previous ?? null;
  }
  return points.reduce((best, p) => Math.abs(p.t - t) < Math.abs(best.t - t) ? p : best);
}

export function TimeChart({ title, domain, series, yMax: rawMax, yFormat, yTicks = 4, reference, markers = [], height = 210 }: {
  title: string; domain: [number, number]; series: Series[]; yMax: number; yFormat: (value: number) => string; yTicks?: number;
  reference?: { y: number; label: string }; markers?: Marker[]; height?: number;
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const yStep = niceStep(rawMax / yTicks);
  const yMax = Math.ceil(rawMax / yStep - 1e-9) * yStep;
  const directLabels = width >= 560;
  const right = M.right + (directLabels ? LABEL_SPACE : 0);
  const innerW = Math.max(0, width - M.left - right);
  const innerH = height - M.top - M.bottom;
  const x = (t: number) => M.left + ((t - domain[0]) / (domain[1] - domain[0])) * innerW;
  const y = (v: number) => M.top + innerH - (Math.min(v, yMax) / yMax) * innerH;
  const visible = useMemo(() => series.map((s) => ({ ...s, points: s.points.filter((p) => p.t >= domain[0] && p.t <= domain[1]) })), [series, domain]);
  const allTimes = useMemo(() => [...new Set(visible.flatMap((s) => s.points.filter((p) => p.y != null).map((p) => p.t)))].sort((a, b) => a - b), [visible]);

  const path = (s: Series) => {
    let d = "";
    let last: Point | null = null;
    for (const p of s.points) {
      if (p.y == null) { last = null; continue; }
      if (!last) d += `M${x(p.t)},${y(p.y)}`;
      else d += s.step ? `H${x(p.t)}V${y(p.y)}` : `L${x(p.t)},${y(p.y)}`;
      last = p;
    }
    if (s.step && last) d += `H${x(domain[1])}`;
    return d;
  };

  const onMove = (event: React.PointerEvent<SVGRectElement>) => {
    if (!allTimes.length) return;
    const box = event.currentTarget.getBoundingClientRect();
    const t = domain[0] + ((event.clientX - box.left) / box.width) * (domain[1] - domain[0]);
    setHover(allTimes.reduce((best, candidate) => Math.abs(candidate - t) < Math.abs(best - t) ? candidate : best));
  };

  const ticks = timeTicks(domain, innerW);
  const empty = allTimes.length === 0;
  const endLabels = visible.map((s) => {
    const last = s.points.filter((p) => p.y != null).at(-1);
    return last ? { s, yPos: y(last.y as number), value: last.y as number } : null;
  }).filter((item): item is { s: Series; yPos: number; value: number } => item != null).sort((a, b) => a.yPos - b.yPos);
  for (let i = 1; i < endLabels.length; i++) endLabels[i].yPos = Math.max(endLabels[i].yPos, endLabels[i - 1].yPos + 26);

  return <figure className="tchart">
    <figcaption><span>{title}</span>{visible.length > 1 && <span className="tchart-legend">{visible.map((s) => <span key={s.key}><i style={{ borderColor: s.color }} />{s.label}</span>)}{reference && <span><i className="ref" />{reference.label}</span>}</span>}</figcaption>
    <div className="tchart-body" ref={ref} style={{ height }}>
      {width > 0 && <svg width={width} height={height} role="img" aria-label={title}>
        {Array.from({ length: Math.round(yMax / yStep) + 1 }, (_, i) => yStep * i).map((v) => <g key={v}><line x1={M.left} x2={M.left + innerW} y1={y(v)} y2={y(v)} className="grid" /><text x={M.left - 8} y={y(v)} className="axis" textAnchor="end" dominantBaseline="middle">{yFormat(v)}</text></g>)}
        {ticks.map((tick) => <text key={tick.t} x={x(tick.t)} y={height - 8} className="axis" textAnchor="middle">{tick.label}</text>)}
        {reference && <line x1={M.left} x2={M.left + innerW} y1={y(reference.y)} y2={y(reference.y)} className="ref-line" />}
        {visible.map((s) => <path key={s.key} d={path(s)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />)}
        {visible.filter((s) => !s.step && s.points.length <= 60).map((s) => s.points.filter((p) => p.y != null).map((p) => <circle key={`${s.key}-${p.t}`} cx={x(p.t)} cy={y(p.y as number)} r={4} fill={s.color} className="dot-ring" />))}
        {markers.filter((m) => m.t >= domain[0] && m.t <= domain[1]).map((m) => <g key={`m-${m.t}`} transform={`translate(${x(m.t)},${y(m.y) - 12})`}><path d="M0,-6L6,5H-6Z" className="alert-mark" /></g>)}
        {directLabels && endLabels.map(({ s, yPos, value }) => <g key={`l-${s.key}`}><text x={M.left + innerW + 10} y={yPos - 3} className="direct-value">{s.format(value)}</text><text x={M.left + innerW + 10} y={yPos + 10} className="direct-label">{s.label}</text></g>)}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={M.top + innerH} className="crosshair" />}
        {hover != null && visible.map((s) => { const p = valueAt(s, hover); return p && p.y != null ? <circle key={`h-${s.key}`} cx={x(p.t === hover || !s.step ? p.t : hover)} cy={y(p.y)} r={5} fill={s.color} className="dot-ring" /> : null; })}
        {empty && <text x={M.left + innerW / 2} y={M.top + innerH / 2} className="axis" textAnchor="middle">Sin datos en este rango</text>}
        <rect x={M.left} y={M.top} width={innerW} height={innerH} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>}
      {hover != null && <div className="tooltip" style={{ left: Math.min(x(hover) + 12, width - 190), top: M.top }}>
        <small>{tooltipDate.format(new Date(hover))}</small>
        {visible.map((s) => { const p = valueAt(s, hover); return <div key={s.key}><i style={{ borderColor: s.color }} /><b>{p?.y != null ? s.format(p.y) : "—"}</b><span>{s.label}</span></div>; })}
        {markers.some((m) => m.t === hover) && <div className="tooltip-alert">△ {markers.find((m) => m.t === hover)?.label}</div>}
      </div>}
    </div>
  </figure>;
}

export function RetrainRug({ domain, events }: { domain: [number, number]; events: RugEvent[] }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<RugEvent | null>(null);
  const right = M.right + (width >= 560 ? LABEL_SPACE : 0);
  const innerW = Math.max(0, width - M.left - right);
  const x = (t: number) => M.left + ((t - domain[0]) / (domain[1] - domain[0])) * innerW;
  const visible = events.filter((e) => e.t >= domain[0] && e.t <= domain[1]);
  return <figure className="tchart rug">
    <figcaption><span>Reentrenamientos · {visible.length}</span><span className="tchart-legend"><span><i className="tick" />Modelo reentrenado</span><span><i className="tick red" />Drift recomendó reentrenar</span></span></figcaption>
    <div className="tchart-body" ref={ref} style={{ height: 34 }}>
      {width > 0 && <svg width={width} height={34} role="img" aria-label={`${visible.length} reentrenamientos en el rango`}>
        <line x1={M.left} x2={M.left + innerW} y1={30} y2={30} className="grid" />
        {visible.map((e) => <g key={e.t} onPointerEnter={() => setHover(e)} onPointerLeave={() => setHover(null)}>
          <line x1={x(e.t)} x2={x(e.t)} y1={e.tone === "red" ? 4 : 10} y2={30} className={`rug-tick ${e.tone}`} />
          <rect x={x(e.t) - 6} y={0} width={12} height={34} fill="transparent" />
        </g>)}
      </svg>}
      {hover && <div className="tooltip" style={{ left: Math.min(x(hover.t) + 12, width - 190), top: 36 }}><small>{tooltipDate.format(new Date(hover.t))}</small><div><b>{hover.label}</b></div></div>}
    </div>
  </figure>;
}

export function Histogram({ title, values, current, currentLabel, format, color, unit, bins = 12 }: {
  title: string; values: number[]; current: number | null; currentLabel: string; format: (value: number) => string; color: string; unit: string; bins?: number;
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const height = 132;
  const m = { top: 20, right: 12, bottom: 24, left: 12 };
  const innerW = Math.max(0, width - m.left - m.right);
  const innerH = height - m.top - m.bottom;
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted.at(-1);
  const median = sorted.length ? (sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2) : null;
  // A metric that never changed collapses into one bin instead of an invented spread.
  const flat = min == null || max == null || Math.abs(max - min) <= Math.abs(max) * 1e-9;
  const count = flat ? 1 : bins;
  const lo = flat ? (min ?? 0) : min!;
  const size = flat ? 1 : (max! - min!) / count;
  const buckets = Array.from({ length: sorted.length ? count : 0 }, (_, i) => ({ from: lo + i * size, to: flat ? lo : lo + (i + 1) * size, n: 0 }));
  for (const v of sorted) buckets[flat ? 0 : Math.min(count - 1, Math.floor((v - lo) / size))].n += 1;
  const top = Math.max(1, ...buckets.map((b) => b.n));
  const barW = flat ? Math.min(48, innerW) : innerW / count;
  const barX = (i: number) => flat ? m.left + (innerW - barW) / 2 : m.left + i * barW;
  const valueX = (v: number) => flat ? m.left + innerW / 2 : m.left + ((v - lo) / (max! - lo)) * innerW;
  const bar = (x: number, w: number, h: number) => {
    const r = Math.min(4, w / 2, h);
    const y0 = m.top + innerH;
    return `M${x},${y0}V${y0 - h + r}Q${x},${y0 - h} ${x + r},${y0 - h}H${x + w - r}Q${x + w},${y0 - h} ${x + w},${y0 - h + r}V${y0}Z`;
  };
  const hovered = hover != null ? buckets[hover] : null;
  return <figure className="hist">
    <figcaption><span>{title}</span><small>{sorted.length ? `n=${sorted.length} · mediana ${format(median!)}` : "Sin datos"}</small></figcaption>
    <div className="tchart-body" ref={ref} style={{ height }}>
      {width > 0 && <svg width={width} height={height} role="img" aria-label={`Distribución de ${title}`}>
        <line x1={m.left} x2={m.left + innerW} y1={m.top + innerH} y2={m.top + innerH} className="grid" />
        {buckets.map((b, i) => { const w = Math.max(1, barW - 2); const h = b.n ? Math.max(2, (b.n / top) * innerH) : 0; return <g key={i} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
          {h > 0 && <path d={bar(barX(i) + 1, w, h)} fill={color} opacity={hover == null || hover === i ? 1 : 0.55} />}
          <rect x={barX(i)} y={m.top} width={barW} height={innerH} fill="transparent" />
        </g>; })}
        {current != null && sorted.length > 0 && <g className="hist-current"><line x1={valueX(current)} x2={valueX(current)} y1={m.top - 4} y2={m.top + innerH} /><text x={Math.min(Math.max(valueX(current), m.left + 40), m.left + innerW - 40)} y={m.top - 8} textAnchor="middle">{currentLabel} {format(current)}</text></g>}
        {sorted.length > 0 && (flat
          ? <text x={m.left + innerW / 2} y={height - 6} className="axis" textAnchor="middle">sin variación</text>
          : <><text x={m.left} y={height - 6} className="axis">{format(min!)}</text><text x={m.left + innerW} y={height - 6} className="axis" textAnchor="end">{format(max!)}</text></>)}
        {!sorted.length && <text x={m.left + innerW / 2} y={m.top + innerH / 2} className="axis" textAnchor="middle">Sin datos en este rango</text>}
      </svg>}
      {hovered && <div className="tooltip" style={{ left: Math.max(0, Math.min(barX(hover!) + barW / 2 - 85, width - 175)), top: 0 }}><small>{flat ? format(hovered.from) : `${format(hovered.from)} – ${format(hovered.to)}`}</small><div><i style={{ borderColor: color }} /><b>{hovered.n}</b><span>{unit}</span></div></div>}
    </div>
  </figure>;
}

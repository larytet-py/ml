#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python dependency. Run this script with the project venv, for example "
        "`.venv/bin/python skills/sobol-gradient-parquet-analysis/scripts/"
        "analyze_sobol_gradient_parquet.py ...`, or install numpy, pandas, pyarrow, and scikit-learn."
    ) from exc


CANONICAL_PARAMS = [
    "param__roc_window_size",
    "param__vol_window_size",
    "param__roc_threshold",
    "param__vol_threshold",
]

REQUIRED_METRICS = [
    "metric__total",
    "metric__itm_expiries",
    "metric__total_pnl",
    "metric__avg_pnl",
    "metric__max_drawdown",
]

RANK_COLUMNS = [
    "metric__total_pnl",
    "metric__itm_expiries",
    "metric__max_drawdown",
    "metric__total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze profitable Sobol-gradient option-strategy regions from a trials parquet."
    )
    parser.add_argument("trials_parquet", help="Path to the trials parquet file.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated outputs. Defaults to parquet parent.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to '<parquet-stem>_region_analysis'.")
    parser.add_argument("--min-trades", type=float, default=3.0, help="Good-run threshold: metric__total must be greater than this.")
    parser.add_argument("--max-itm", type=float, default=2.0, help="Good-run threshold: metric__itm_expiries must be less than this.")
    parser.add_argument(
        "--allow-nonpositive-pnl",
        action="store_true",
        help="Do not require metric__total_pnl > 0 for good runs.",
    )
    parser.add_argument("--seed-top-ratio", type=float, default=0.30, help="Source-style seed ratio to summarize.")
    parser.add_argument("--top-ratio", type=float, default=0.05, help="Top-ranked subset ratio to summarize.")
    parser.add_argument(
        "--eps",
        default="0.02,0.025,0.03,0.04,0.06,0.08,0.10,0.12",
        help="Comma-separated normalized Chebyshev eps values for connectivity sweeps.",
    )
    parser.add_argument("--fine-eps", type=float, default=0.02, help="Epsilon used for detailed component CSVs.")
    parser.add_argument("--sample-other-good", type=int, default=12000, help="Good rows outside the seed set sampled into HTML.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-html", action="store_true", help="Skip self-contained HTML projection.")
    return parser.parse_args()


def parse_eps_values(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--eps must contain at least one value")
    return values


def short_name(column: str) -> str:
    return column.replace("param__", "").replace("metric__", "")


def detect_param_columns(df: pd.DataFrame) -> List[str]:
    params = [c for c in df.columns if c.startswith("param__")]
    if not params:
        raise ValueError("No param__ columns found in parquet.")
    canonical = [c for c in CANONICAL_PARAMS if c in params]
    extras = [c for c in params if c not in canonical]
    return canonical + extras


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required parquet columns: {missing}")


def normalize_frame(sub: pd.DataFrame, param_cols: List[str], bounds: pd.DataFrame) -> np.ndarray:
    if sub.empty:
        return np.zeros((0, len(param_cols)), dtype=float)
    mins = bounds.loc["min", param_cols]
    widths = bounds.loc["max", param_cols] - mins
    widths = widths.replace(0.0, np.nan)
    out = (sub[param_cols] - mins) / widths
    out = out.fillna(0.0).clip(0.0, 1.0)
    return out.to_numpy(dtype=float)


def cluster_labels(sub: pd.DataFrame, param_cols: List[str], bounds: pd.DataFrame, eps: float) -> np.ndarray:
    x = normalize_frame(sub, param_cols, bounds)
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    return DBSCAN(eps=float(eps), min_samples=1, metric="chebyshev", n_jobs=-1).fit_predict(x)


def connectivity_table(sub: pd.DataFrame, param_cols: List[str], bounds: pd.DataFrame, eps_values: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = int(len(sub))
    for eps in eps_values:
        labels = cluster_labels(sub, param_cols, bounds, eps)
        vals, counts = np.unique(labels, return_counts=True)
        order = np.argsort(counts)[::-1]
        largest = int(counts[order[0]]) if len(counts) else 0
        top5 = int(np.sum(counts[order[:5]])) if len(counts) else 0
        top10 = int(np.sum(counts[order[:10]])) if len(counts) else 0
        out.append(
            {
                "eps": float(eps),
                "components": int(len(vals)),
                "largest_n": largest,
                "largest_share": float(largest / n) if n else 0.0,
                "top5_components_n": top5,
                "top5_components_share": float(top5 / n) if n else 0.0,
                "top10_components_n": top10,
                "top10_components_share": float(top10 / n) if n else 0.0,
            }
        )
    return out


def component_summary(sub: pd.DataFrame, param_cols: List[str], bounds: pd.DataFrame, eps: float) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame()
    labels = cluster_labels(sub, param_cols, bounds, eps)
    h = sub.copy()
    h["component"] = labels
    rows: List[Dict[str, Any]] = []
    for comp, chunk in h.groupby("component", sort=False):
        row: Dict[str, Any] = {
            "component": int(comp),
            "n": int(len(chunk)),
            "pnl_max": float(chunk["metric__total_pnl"].max()),
            "pnl_median": float(chunk["metric__total_pnl"].median()),
            "pnl_mean": float(chunk["metric__total_pnl"].mean()),
            "avg_pnl_mean": float(chunk["metric__avg_pnl"].mean()),
            "avg_pnl_std": float(chunk["metric__avg_pnl"].std(ddof=0)),
            "avg_pnl_iqr": float(chunk["metric__avg_pnl"].quantile(0.75) - chunk["metric__avg_pnl"].quantile(0.25)),
            "avg_pnl_min": float(chunk["metric__avg_pnl"].min()),
            "avg_pnl_median": float(chunk["metric__avg_pnl"].median()),
            "avg_pnl_max": float(chunk["metric__avg_pnl"].max()),
            "trades_min": float(chunk["metric__total"].min()),
            "trades_median": float(chunk["metric__total"].median()),
            "trades_max": float(chunk["metric__total"].max()),
            "itm_median": float(chunk["metric__itm_expiries"].median()),
            "drawdown_median": float(chunk["metric__max_drawdown"].median()),
        }
        for param in param_cols:
            name = short_name(param)
            row[f"{name}_min"] = float(chunk[param].min())
            row[f"{name}_median"] = float(chunk[param].median())
            row[f"{name}_max"] = float(chunk[param].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n", "pnl_max"], ascending=False).reset_index(drop=True)


def ratio_count(n: int, ratio: float) -> int:
    if n <= 0:
        return 0
    return max(1, int(math.ceil(n * float(ratio))))


def top_ratio_label(ratio: float) -> str:
    pct = int(round(float(ratio) * 100))
    return f"top{pct:02d}"


def finite_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        return 0.0
    return out


def write_html(
    *,
    path: Path,
    ranked: pd.DataFrame,
    labels: np.ndarray,
    param_cols: List[str],
    bounds: pd.DataFrame,
    seed_n: int,
    top_n: int,
    sample_other_good: int,
    random_state: int,
    summary: Dict[str, Any],
) -> bool:
    if any(c not in param_cols for c in CANONICAL_PARAMS):
        print("[html] skipped: canonical four Sobol-gradient param columns were not all present")
        return False

    ranked = ranked.copy()
    ranked["rank_bucket"] = "good"
    ranked.loc[ranked.index < seed_n, "rank_bucket"] = "source_top"
    ranked.loc[ranked.index < top_n, "rank_bucket"] = "top"
    labels_df = ranked[CANONICAL_PARAMS].copy()
    labels_df["fine_component"] = labels
    source_rows = ranked.iloc[:seed_n]
    others = ranked.iloc[seed_n:]
    if sample_other_good > 0 and len(others) > sample_other_good:
        others = others.sample(n=sample_other_good, random_state=random_state)
    viz = pd.concat([source_rows, others], ignore_index=True).merge(labels_df, on=CANONICAL_PARAMS, how="left")

    fields = CANONICAL_PARAMS + REQUIRED_METRICS + ["rank_bucket", "fine_component"]
    records: List[Dict[str, Any]] = []
    for item in viz[fields].to_dict(orient="records"):
        record: Dict[str, Any] = {}
        for key, value in item.items():
            out_key = short_name(key)
            if isinstance(value, (int, np.integer)):
                record[out_key] = int(value)
            elif isinstance(value, (float, np.floating)):
                record[out_key] = round(finite_float(value), 4)
            else:
                record[out_key] = value
        records.append(record)

    html_summary = {
        "rows_total": int(summary["rows_total"]),
        "unique_rows": int(summary["unique_parameter_rows"]),
        "good_rows": int(summary["good_unique_rows"]),
        "seed_rows": int(summary["source_top_seed_rows"]),
        "top_rows": int(summary["top_ranked_rows"]),
        "best_total_pnl": finite_float(ranked.iloc[0]["metric__total_pnl"]) if len(ranked) else 0.0,
        "bounds": {
            short_name(c): {"min": finite_float(bounds.loc["min", c]), "max": finite_float(bounds.loc["max", c])}
            for c in CANONICAL_PARAMS
        },
    }

    html = HTML_TEMPLATE.replace("__SUMMARY__", json.dumps(html_summary, separators=(",", ":")))
    html = html.replace("__POINTS__", json.dumps(records, separators=(",", ":")))
    path.write_text(html)
    return True


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sobol Gradient Region Projections</title>
<style>
body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:#f7f7f4;color:#222}
header{padding:18px 24px 10px;border-bottom:1px solid #d9d7cf;background:#fff;position:sticky;top:0;z-index:2}
h1{font-size:22px;margin:0 0 6px;font-weight:700;letter-spacing:0}
.summary{display:flex;flex-wrap:wrap;gap:10px 18px;font-size:13px;color:#454545}
main{padding:18px 24px 28px}
.controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:0 0 12px;font-size:13px}
select{font:inherit;padding:4px 8px;border:1px solid #c8c4b8;border-radius:4px;background:white}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}
section{background:#fff;border:1px solid #d9d7cf;border-radius:6px;padding:12px}
h2{font-size:15px;margin:0 0 8px}
canvas{width:100%;height:420px;display:block;background:#fbfbf8;border:1px solid #e4e1d8;border-radius:4px}
.legend{font-size:12px;color:#555;margin-top:8px}
code{background:#efeee8;padding:1px 4px;border-radius:3px}
</style>
</head>
<body>
<header>
<h1>Sobol Gradient Good-Run Region Projections</h1>
<div class="summary" id="summary"></div>
</header>
<main>
<div class="controls">
<label>Dataset <select id="dataset"><option value="all">good plus sample</option><option value="source_top">source top seed set</option><option value="top">top-ranked subset</option></select></label>
<label>Color <select id="colorMetric"><option value="total_pnl">total_pnl</option><option value="avg_pnl">avg_pnl</option><option value="total">trades</option></select></label>
</div>
<div class="grid">
<section><h2>Total-PnL band: x=roc_window, y=vol_window, z=vol_threshold; roc_threshold -0.10..0.08</h2><canvas id="c1" width="900" height="520"></canvas><div class="legend">Use this slice for the dominant high-total-PnL basin.</div></section>
<section><h2>High avg-PnL shelf: x=roc_window, y=vol_window, z=vol_threshold; roc_threshold 0.03..0.16</h2><canvas id="c2" width="900" height="520"></canvas><div class="legend">Use this slice for fewer-trade, higher-average shelves.</div></section>
<section><h2>Threshold shape: x=roc_window, y=vol_window, z=roc_threshold; vol_threshold 0.39..0.44</h2><canvas id="c3" width="900" height="520"></canvas><div class="legend">Narrow vol-threshold band near the best total-PnL rows.</div></section>
<section><h2>Threshold shelf: x=roc_window, y=vol_window, z=roc_threshold; vol_threshold 0.47..0.56</h2><canvas id="c4" width="900" height="520"></canvas><div class="legend">Broader high-average shelf at higher vol thresholds.</div></section>
<section><h2>2D heatmap: roc_window vs vol_window</h2><canvas id="c5" width="900" height="520"></canvas><div class="legend">Cell color is median selected metric.</div></section>
<section><h2>2D heatmap: roc_threshold vs vol_threshold</h2><canvas id="c6" width="900" height="520"></canvas><div class="legend">Shows the threshold footprint of the good-run set.</div></section>
</div>
</main>
<script>
const SUMMARY=__SUMMARY__;
const POINTS=__POINTS__;
const BOUNDS=SUMMARY.bounds;
const PALETTE=[[45,59,135],[35,126,167],[45,156,117],[238,183,62],[196,72,52]];
function lerp(a,b,t){return a+(b-a)*t}
function colorScale(v,lo,hi){let t=hi>lo?(v-lo)/(hi-lo):0.5;t=Math.max(0,Math.min(1,t));const s=t*(PALETTE.length-1);const i=Math.min(PALETTE.length-2,Math.floor(s));const f=s-i;const a=PALETTE[i],b=PALETTE[i+1];return `rgb(${Math.round(lerp(a[0],b[0],f))},${Math.round(lerp(a[1],b[1],f))},${Math.round(lerp(a[2],b[2],f))})`}
function dataset(){const ds=document.getElementById("dataset").value;if(ds==="top")return POINTS.filter(p=>p.rank_bucket==="top");if(ds==="source_top")return POINTS.filter(p=>p.rank_bucket==="top"||p.rank_bucket==="source_top");return POINTS}
function norm(v,k){const b=BOUNDS[k];return (v-b.min)/(b.max-b.min)}
function project(x,y,z,w,h){return [w*0.50+(x-0.5)*w*0.62-(y-0.5)*w*0.36,h*0.70+(x-0.5)*h*0.18+(y-0.5)*h*0.34-z*h*0.58]}
function metricRange(points,m){const vals=points.map(p=>p[m]).filter(Number.isFinite).sort((a,b)=>a-b);if(!vals.length)return[0,1];const q=r=>vals[Math.max(0,Math.min(vals.length-1,Math.floor(r*(vals.length-1))))];return[q(0.02),q(0.98)]}
function axes(ctx,w,h,labels){ctx.save();ctx.strokeStyle="#777";ctx.fillStyle="#333";ctx.font="12px system-ui";const o=project(0,0,0,w,h),x=project(1,0,0,w,h),y=project(0,1,0,w,h),z=project(0,0,1,w,h);[[o,x,labels[0]],[o,y,labels[1]],[o,z,labels[2]]].forEach(([a,b,l])=>{ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.fillText(l,b[0]+5,b[1]-5)});ctx.restore()}
function draw3d(id,filter,keys){const c=document.getElementById(id),ctx=c.getContext("2d"),w=c.width,h=c.height,m=document.getElementById("colorMetric").value;ctx.clearRect(0,0,w,h);ctx.fillStyle="#fbfbf8";ctx.fillRect(0,0,w,h);let pts=dataset().filter(filter);const [lo,hi]=metricRange(pts,m);pts=pts.map(p=>{const q=project(norm(p[keys[0]],keys[0]),norm(p[keys[1]],keys[1]),norm(p[keys[2]],keys[2]),w,h);return{p,sx:q[0],sy:q[1],d:norm(p[keys[1]],keys[1])+norm(p[keys[2]],keys[2])}}).sort((a,b)=>a.d-b.d);axes(ctx,w,h,keys);ctx.globalAlpha=.72;for(const q of pts){ctx.fillStyle=colorScale(q.p[m],lo,hi);const r=q.p.rank_bucket==="top"?3:q.p.rank_bucket==="source_top"?2.2:1.4;ctx.beginPath();ctx.arc(q.sx,q.sy,r,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;ctx.fillStyle="#333";ctx.font="12px system-ui";ctx.fillText(`${pts.length} points, color=${m} [${lo.toFixed(1)}, ${hi.toFixed(1)}]`,14,22)}
function drawHeat(id,xKey,yKey){const c=document.getElementById(id),ctx=c.getContext("2d"),w=c.width,h=c.height,m=document.getElementById("colorMetric").value;ctx.clearRect(0,0,w,h);ctx.fillStyle="#fbfbf8";ctx.fillRect(0,0,w,h);const pts=dataset();const xb=xKey.includes("threshold")?42:Math.round(BOUNDS[xKey].max-BOUNDS[xKey].min+1);const yb=yKey.includes("threshold")?42:Math.round(BOUNDS[yKey].max-BOUNDS[yKey].min+1);const cells=new Map();for(const p of pts){const xi=Math.max(0,Math.min(xb-1,Math.floor(norm(p[xKey],xKey)*xb)));const yi=Math.max(0,Math.min(yb-1,Math.floor(norm(p[yKey],yKey)*yb)));const k=xi+","+yi;if(!cells.has(k))cells.set(k,[]);cells.get(k).push(p[m])}const meds=[];for(const arr of cells.values()){arr.sort((a,b)=>a-b);meds.push(arr[Math.floor(arr.length/2)])}const lo=Math.min(...meds),hi=Math.max(...meds);const pad=54,pw=w-pad-18,ph=h-pad-28,cw=pw/xb,ch=ph/yb;for(const [k,arr] of cells){const [xi,yi]=k.split(",").map(Number);const v=arr[Math.floor(arr.length/2)];ctx.fillStyle=colorScale(v,lo,hi);ctx.globalAlpha=Math.min(.95,.25+Math.sqrt(arr.length)/10);ctx.fillRect(pad+xi*cw,h-pad-(yi+1)*ch,Math.max(1,cw),Math.max(1,ch))}ctx.globalAlpha=1;ctx.strokeStyle="#777";ctx.strokeRect(pad,h-pad-ph,pw,ph);ctx.fillStyle="#333";ctx.font="12px system-ui";ctx.fillText(`${pts.length} points, median ${m} by cell`,14,22);ctx.fillText(xKey,pad+pw/2-35,h-18);ctx.save();ctx.translate(16,h-pad-ph/2+35);ctx.rotate(-Math.PI/2);ctx.fillText(yKey,0,0);ctx.restore()}
function drawAll(){draw3d("c1",p=>p.roc_threshold>=-.10&&p.roc_threshold<=.08,["roc_window_size","vol_window_size","vol_threshold"]);draw3d("c2",p=>p.roc_threshold>=.03&&p.roc_threshold<=.16,["roc_window_size","vol_window_size","vol_threshold"]);draw3d("c3",p=>p.vol_threshold>=.39&&p.vol_threshold<=.44,["roc_window_size","vol_window_size","roc_threshold"]);draw3d("c4",p=>p.vol_threshold>=.47&&p.vol_threshold<=.56,["roc_window_size","vol_window_size","roc_threshold"]);drawHeat("c5","roc_window_size","vol_window_size");drawHeat("c6","roc_threshold","vol_threshold")}
document.getElementById("summary").innerHTML=`<span>Rows: <code>${SUMMARY.rows_total.toLocaleString()}</code></span><span>Unique params: <code>${SUMMARY.unique_rows.toLocaleString()}</code></span><span>Good rows: <code>${SUMMARY.good_rows.toLocaleString()}</code></span><span>Seed set: <code>${SUMMARY.seed_rows.toLocaleString()}</code></span><span>Top rows: <code>${SUMMARY.top_rows.toLocaleString()}</code></span><span>Best total_pnl: <code>${SUMMARY.best_total_pnl.toFixed(2)}</code></span>`;
document.getElementById("dataset").addEventListener("change",drawAll);
document.getElementById("colorMetric").addEventListener("change",drawAll);
drawAll();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.trials_parquet)
    output_dir = Path(args.output_dir) if args.output_dir else parquet_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{parquet_path.stem}_region_analysis"
    eps_values = parse_eps_values(args.eps)

    df = pd.read_parquet(parquet_path)
    param_cols = detect_param_columns(df)
    require_columns(df, param_cols + REQUIRED_METRICS)

    rows_total = int(len(df))
    duplicate_parameter_rows = int(df.duplicated(param_cols).sum())
    unique_df = df.drop_duplicates(param_cols, keep="first").copy()
    bounds = unique_df[param_cols].agg(["min", "max"])

    good_mask = (
        (unique_df["metric__total"] > float(args.min_trades))
        & (unique_df["metric__itm_expiries"] < float(args.max_itm))
    )
    if not args.allow_nonpositive_pnl:
        good_mask &= unique_df["metric__total_pnl"] > 0.0
    good = unique_df.loc[good_mask].copy()
    ranked = good.sort_values(RANK_COLUMNS, ascending=[False, True, False, False]).reset_index(drop=True)

    seed_n = ratio_count(len(ranked), args.seed_top_ratio)
    top_n = ratio_count(len(ranked), args.top_ratio)
    seeds = ranked.head(seed_n).copy()
    top = ranked.head(top_n).copy()

    good_components = component_summary(ranked, param_cols, bounds, args.fine_eps)
    seed_components = component_summary(seeds, param_cols, bounds, args.fine_eps)
    top_components = component_summary(top, param_cols, bounds, args.fine_eps)

    eps_label = str(args.fine_eps).replace(".", "")
    seed_label = top_ratio_label(args.seed_top_ratio)
    top_label = top_ratio_label(args.top_ratio)
    good_csv = output_dir / f"{prefix}_components_good_eps{eps_label}.csv"
    seed_csv = output_dir / f"{prefix}_components_seed_{seed_label}_eps{eps_label}.csv"
    top_csv = output_dir / f"{prefix}_components_{top_label}_eps{eps_label}.csv"
    good_components.to_csv(good_csv, index=False)
    seed_components.to_csv(seed_csv, index=False)
    top_components.to_csv(top_csv, index=False)

    summary: Dict[str, Any] = {
        "source_parquet": str(parquet_path),
        "rows_total": rows_total,
        "unique_parameter_rows": int(len(unique_df)),
        "duplicate_parameter_rows": duplicate_parameter_rows,
        "good_criteria": {
            "metric__total_gt": float(args.min_trades),
            "metric__itm_expiries_lt": float(args.max_itm),
            "metric__total_pnl_gt": None if args.allow_nonpositive_pnl else 0.0,
        },
        "good_unique_rows": int(len(ranked)),
        "source_rank_sort": [
            "metric__total_pnl desc",
            "metric__itm_expiries asc",
            "metric__max_drawdown desc",
            "metric__total desc",
        ],
        "source_top_ratio": float(args.seed_top_ratio),
        "source_top_seed_rows": int(len(seeds)),
        "top_ratio": float(args.top_ratio),
        "top_ranked_rows": int(len(top)),
        "top_total_pnl_cutoff": finite_float(top.iloc[-1]["metric__total_pnl"]) if len(top) else 0.0,
        "parameter_bounds": {
            short_name(c): {"min": finite_float(bounds.loc["min", c]), "max": finite_float(bounds.loc["max", c])}
            for c in param_cols
        },
        "connectivity": {
            "good_all": connectivity_table(ranked, param_cols, bounds, eps_values),
            f"source_seed_{seed_label}": connectivity_table(seeds, param_cols, bounds, eps_values),
            top_label: connectivity_table(top, param_cols, bounds, eps_values),
        },
        "major_components_fine_eps": {
            "eps": float(args.fine_eps),
            "good_all": good_components.head(12).to_dict(orient="records"),
            f"source_seed_{seed_label}": seed_components.head(12).to_dict(orient="records"),
            top_label: top_components.head(12).to_dict(orient="records"),
        },
        "outputs": {
            "good_components_csv": str(good_csv),
            "seed_components_csv": str(seed_csv),
            "top_components_csv": str(top_csv),
        },
    }

    fine_labels = cluster_labels(ranked, param_cols, bounds, args.fine_eps)
    if not args.no_html:
        html_path = output_dir / f"{prefix}_projection.html"
        if write_html(
            path=html_path,
            ranked=ranked,
            labels=fine_labels,
            param_cols=param_cols,
            bounds=bounds,
            seed_n=seed_n,
            top_n=top_n,
            sample_other_good=int(args.sample_other_good),
            random_state=int(args.random_state),
            summary=summary,
        ):
            summary["outputs"]["projection_html"] = str(html_path)

    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))

    print(f"rows_total={rows_total}")
    print(f"unique_parameter_rows={len(unique_df)}")
    print(f"good_unique_rows={len(ranked)}")
    print(f"source_top_seed_rows={len(seeds)}")
    print(f"top_ranked_rows={len(top)}")
    print(f"wrote {summary_path}")
    print(f"wrote {good_csv}")
    print(f"wrote {seed_csv}")
    print(f"wrote {top_csv}")
    if "projection_html" in summary["outputs"]:
        print(f"wrote {summary['outputs']['projection_html']}")


if __name__ == "__main__":
    main()

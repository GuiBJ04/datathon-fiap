"""
Detecção de drift e dashboard de monitoramento.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats


def detect_data_drift(reference, production, threshold=0.05):
    """Kolmogorov-Smirnov por feature."""
    results = {}
    drifted = []
    common = [c for c in reference.columns if c in production.columns]

    for col in common:
        ref = reference[col].dropna()
        prod = production[col].dropna()
        if len(ref) < 2 or len(prod) < 2:
            continue
        stat, pval = stats.ks_2samp(ref, prod)
        has_drift = pval < threshold
        results[col] = {"ks_stat": round(stat, 4), "p_value": round(pval, 4),
                        "drift": has_drift, "ref_mean": round(ref.mean(), 4),
                        "prod_mean": round(prod.mean(), 4)}
        if has_drift:
            drifted.append(col)

    results["_summary"] = {"total": len(common), "drifted": len(drifted),
                           "features": drifted, "timestamp": datetime.now().isoformat()}
    return results


def generate_dashboard(drift_results, output_path="monitoring/dashboard.html"):
    """Gera dashboard HTML."""
    summary = drift_results.get("_summary", {})
    features = {k: v for k, v in drift_results.items() if k != "_summary"}

    rows = ""
    for feat, d in sorted(features.items(), key=lambda x: x[1].get("p_value", 1)):
        cls = "drift" if d.get("drift") else "ok"
        label = "⚠️ DRIFT" if d.get("drift") else "✅ OK"
        rows += f'<tr class="{cls}"><td>{feat}</td><td>{d["ks_stat"]}</td><td>{d["p_value"]}</td><td>{d["ref_mean"]}</td><td>{d["prod_mean"]}</td><td>{label}</td></tr>'

    status = "⚠️ DRIFT" if summary.get("drifted", 0) > 0 else "✅ SAUDÁVEL"
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Drift Monitor</title>
<style>body{{font-family:system-ui;background:#0f172a;color:#e2e8f0;padding:2rem}}
h1{{color:#38bdf8}}table{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden}}
th{{background:#334155;padding:10px 14px;text-align:left;color:#94a3b8;font-size:.85rem}}
td{{padding:10px 14px;border-bottom:1px solid #334155}}tr.drift{{background:rgba(248,113,113,.1)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0}}
.card{{background:#1e293b;border-radius:10px;padding:1.2rem;border:1px solid #334155}}
.card h3{{color:#94a3b8;font-size:.8rem;text-transform:uppercase;margin-bottom:.4rem}}
.card .v{{font-size:1.8rem;font-weight:700}}.green{{color:#4ade80}}.red{{color:#f87171}}</style></head>
<body><h1>🔍 Monitoramento de Drift</h1>
<div class="cards"><div class="card"><h3>Features</h3><div class="v">{summary.get("total",0)}</div></div>
<div class="card"><h3>Com Drift</h3><div class="v {'red' if summary.get('drifted',0)>0 else 'green'}">{summary.get("drifted",0)}</div></div>
<div class="card"><h3>Status</h3><div class="v {'red' if summary.get('drifted',0)>0 else 'green'}">{status}</div></div></div>
<table><thead><tr><th>Feature</th><th>KS Stat</th><th>P-Value</th><th>Média (Ref)</th><th>Média (Prod)</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table>
<p style="color:#64748b;margin-top:1.5rem;text-align:center">Gerado em {datetime.now().strftime("%Y-%m-%d %H:%M")}</p></body></html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

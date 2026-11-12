"""Self-contained HTML scorecard for a campaign's runs."""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from vivid2.scoring import DEFAULT_WEIGHTS, aggregate, bootstrap_ci, scaling, stack_label

LAYERS = list(DEFAULT_WEIGHTS) + ["overall"]

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--bar:#2f6fdf;--bad:#c2410c;--good:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--fg:#ecebe6;--muted:#a3a29c;--line:#34332f;--card:#1e1e1c;--bar:#6b9cf0;--bad:#fb923c;--good:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 12px}
.muted{color:var(--muted)}.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}.bar{position:relative;min-width:110px}
.bar span{position:absolute;left:0;top:25%;height:50%;background:var(--bar);opacity:.18;border-radius:2px}.bar b{position:relative;font-weight:500}
.ci{color:var(--muted);font-size:12px}.ok{color:var(--good)}.fail{color:var(--bad)}code{font-size:12px}
"""


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def _score_cell(ci: Dict[str, Any]) -> str:
    if ci["mean"] is None:
        return '<td class="num muted">n/a</td>'
    pct = max(0.0, min(100.0, ci["mean"]))
    rng = f'<div class="ci">{ci["lo"]:.0f}–{ci["hi"]:.0f}</div>' if ci["n"] > 1 else ""
    return f'<td class="num bar"><span style="width:{pct:.0f}%"></span><b>{ci["mean"]:.1f}</b>{rng}</td>'


def _open_loop_section(results: List[Dict[str, Any]]) -> str:
    """One row per model evaluated offline on saved frames (reasoning layer only)."""
    if not results:
        return ""
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_agent.setdefault(r["agent"], []).append(r)
    rows = []
    for agent, rs in sorted(by_agent.items(), key=lambda kv: -(bootstrap_ci(r["scores"]["reasoning"] for r in kv[1])["mean"] or 0)):
        ms = [r["metrics"]["reasoning"] for r in rs if r["metrics"]["reasoning"]]
        n = sum(m["decisions"] for m in ms)
        conf = {k: sum(m["confusion"][k] for m in ms) for k in ("tp", "fp", "tn", "fn")}
        prec = conf["tp"] / (conf["tp"] + conf["fp"]) if conf["tp"] + conf["fp"] else None
        rec = conf["tp"] / (conf["tp"] + conf["fn"]) if conf["tp"] + conf["fn"] else None
        hall = conf["fp"] / (conf["fp"] + conf["tn"]) if conf["fp"] + conf["tn"] else None
        err = sum(m["errors"] for m in ms) / n if n else None
        lat = sum(m["latency_mean_s"] * m["decisions"] for m in ms) / n if n else None
        cost = sum(m["cost_usd"] for m in ms)
        rows.append(
            f"<tr><td><b>{html.escape(agent)}</b></td><td class='num'>{len(rs)}</td><td class='num'>{n}</td>"
            f"{_score_cell(bootstrap_ci(r['scores']['reasoning'] for r in rs))}"
            f"<td class='num'>{_fmt(prec, 2)}</td><td class='num'>{_fmt(rec, 2)}</td><td class='num'>{_fmt(hall, 3)}</td>"
            f"<td class='num'>{_fmt(err, 3)}</td><td class='num'>{_fmt(lat, 2)}</td><td class='num'>{_fmt(cost, 2)}</td></tr>"
        )
    return f"""<h2>Open-loop VLM comparison</h2>
<div class="wrap"><table><tr><th>Model</th><th>Runs</th><th>Frames</th><th>Reasoning</th><th>Precision</th><th>Recall</th>
<th>Halluc.</th><th>Error rate</th><th>Latency s</th><th>Cost $</th></tr>{''.join(rows)}</table></div>
<p class="muted">Every model assessed the same saved frames (<code>vivid2 eval-frames</code>). Errors are abstentions and scale the
reasoning score down. Latency of local models was measured with Unreal closed (uncontended GPU).</p>"""


def _is_oracle(r: Dict[str, Any]) -> bool:
    """The reference oracle reads ground truth: shown for reference, never ranked (docs/rules.md)."""
    return not r.get("eligible", (r.get("agent_config") or {}).get("perception") != "gt_perception")


def _stack_rows(results: List[Dict[str, Any]]) -> str:
    oracle = {stack_label(r) for r in results if _is_oracle(r)}
    violations: Dict[str, int] = {}
    for r in results:
        violations[stack_label(r)] = violations.get(stack_label(r), 0) + len(r.get("rule_violations") or [])
    rows = []
    ranked = sorted(aggregate(results).items(),
                    key=lambda kv: (kv[0] in oracle, -(kv[1]["scores"]["overall"]["mean"] or 0)))
    for agent, s in ranked:
        cells = "".join(_score_cell(s["scores"][k]) for k in LAYERS)
        note = " <span class='muted'>(oracle, not eligible)</span>" if agent in oracle else ""
        rows.append(
            f"<tr><td><b>{html.escape(agent)}</b>{note}</td><td class='num'>{s['runs']}</td>"
            f"<td class='num'>{100 * s['success_rate']:.0f}%</td><td class='num'>{s['collision_runs']}</td>"
            f"<td class='num'>{violations.get(agent, 0)}</td>{cells}</tr>"
        )
    return "".join(rows)


TEAM_MODES = {"central": "Teams, central (one agent commands every drone)",
              "distributed": "Teams, distributed (one agent per drone, radio only)"}


def _team_sections(results: List[Dict[str, Any]]) -> str:
    """One stack table per team mode, then how each team stack scales with the number of drones."""
    if not results:
        return ""
    head = "".join(f"<th>{k}</th>" for k in LAYERS)
    out = []
    for mode, title in TEAM_MODES.items():
        rs = [r for r in results if r["team"].get("mode") == mode]
        if rs:
            out.append(f"""<h2>{html.escape(title)}</h2>
<div class="wrap"><table><tr><th>Stack x drones</th><th>Runs</th><th>Success</th><th>Collision runs</th><th>Rule violations</th>{head}</tr>{_stack_rows(rs)}</table></div>""")
    rows = []
    for agent, sizes in sorted(scaling(results).items()):
        for n, s in sizes.items():
            rows.append(
                f"<tr><td><b>{html.escape(agent)}</b></td><td class='num'>{n}</td><td class='num'>{s['runs']}</td>"
                f"<td class='num'>{100 * s['success_rate']:.0f}%</td><td class='num'>{_fmt(s['time_to_find_all_s'], 0)}</td>"
                f"<td class='num'>{_fmt(s['time_to_90pct_coverage_s'], 0)}</td><td class='num'>{_fmt(s['coverage_pct'])}</td>"
                f"<td class='num'>{_fmt(s['speedup'], 2)}</td><td class='num'>{_fmt(s['efficiency'], 2)}</td>"
                f"<td class='num'><b>{_fmt(s['overall'])}</b></td></tr>")
    if rows:
        out.append(f"""<h2>Team scaling</h2>
<div class="wrap"><table><tr><th>Stack</th><th>Drones</th><th>Runs</th><th>Success</th><th>Find all s</th><th>90% coverage s</th>
<th>Coverage %</th><th>Speedup</th><th>Efficiency</th><th>Overall</th></tr>{''.join(rows)}</table></div>
<p class="muted">Find all: mission time until every target was found, a target never found counting the whole timeout.
Speedup: one drone's mean over N drones' mean, on the runs both flew (same scenario, settings and seed). Efficiency:
speedup / N (1 = N drones are N times as fast).</p>""")
    return "\n".join(out)


def render(results: List[Dict[str, Any]], title: str) -> str:
    open_loop = [r for r in results if r.get("mode") == "open_loop"]
    results = [r for r in results if r.get("mode") != "open_loop"]
    teams = [r for r in results if r.get("team")]
    singles = [r for r in results if not r.get("team")]
    head = "".join(f"<th>{k}</th>" for k in LAYERS)
    stacks = ('<h2>Stacks</h2>\n<div class="wrap"><table><tr><th>Stack</th><th>Runs</th><th>Success</th>'
              f'<th>Collision runs</th><th>Rule violations</th>{head}</tr>{_stack_rows(singles)}</table></div>') if singles else ""

    run_rows = []
    for r in sorted(results, key=lambda r: (r["scenario_id"], r["agent"], r["run_id"])):
        mi, sa = r["metrics"]["mission"], r["metrics"]["safety"]
        rs = r.get("reasoning_summary") or {}
        status = '<span class="ok">success</span>' if mi["success"] else f'<span class="fail">{_fmt(mi["end_reason"])}</span>'
        run_rows.append(
            "<tr>"
            f"<td><code>{html.escape(r['run_id'])}</code></td><td>{html.escape(stack_label(r))}</td>"
            f"<td><code>{html.escape(json.dumps(r['params']))}</code> seed {r['seed']}</td><td>{status}</td>"
            f"<td class='num'>{mi['targets_found']}/{mi['targets_total']}</td>"
            f"<td class='num'>{_fmt(mi['mean_time_to_find_s'])}</td><td class='num'>{_fmt(mi['mean_localization_error_m'])}</td>"
            f"<td class='num'>{mi['false_alarms']}</td><td class='num'>{sa['collisions']}</td>"
            f"<td class='num'>{_fmt(r['metrics']['planning']['coverage_pct'])}</td>"
            f"<td class='num'>{_fmt(rs.get('cost_usd'), 3)}</td>"
            f"<td class='num'><b>{_fmt(r['scores']['overall'])}</b></td></tr>"
        )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="muted">VIVID 2 scorecard · {len(results)} flown runs{f" · {len(open_loop)} open-loop evaluations" if open_loop else ""} · generated {datetime.now():%Y-%m-%d %H:%M}</div>
{stacks}
{_team_sections(teams)}
<p class="muted">Layer scores are 0–100 means with a 95% bootstrap interval over runs. "n/a" means the stack does not expose that layer.
Overall is the weighted mean of available layers, capped at 25 for any run with a collision (in a team: any drone's).</p>
<h2>Runs</h2>
<div class="wrap"><table><tr><th>Run</th><th>Stack</th><th>Params</th><th>Result</th><th>Found</th><th>TTF s</th><th>Loc err m</th>
<th>False alarms</th><th>Collisions</th><th>Coverage %</th><th>VLM $</th><th>Overall</th></tr>{''.join(run_rows)}</table></div>
{_open_loop_section(open_loop)}
</main></body></html>"""


def load_results(runs_dir: Path) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(runs_dir).rglob("result.json"))]


def write_report(runs_dir: Path, out: Path = None, title: str = None) -> Path:
    runs_dir = Path(runs_dir)
    results = load_results(runs_dir)
    if not results:
        raise FileNotFoundError(f"no result.json files under {runs_dir}")
    out = Path(out) if out else runs_dir / "scorecard.html"
    out.write_text(render(results, title or f"VIVID 2 — {runs_dir.name}"), encoding="utf-8")
    return out

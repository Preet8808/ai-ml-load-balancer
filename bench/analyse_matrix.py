"""
Phase 8 analysis of the Phase 7 matrix.

Reads `scratch/matrix/results.jsonl` (one row per completed cell) and the raw
Parquet files under `scratch/matrix/raw/`, and produces:

  * per-scenario tables: mean p95 with a bootstrap 95% CI, throughput, errors,
    SLA violations and Jain fairness
  * the plan's hypothesis tests (H1-H4) on run-level statistics, with Welch's
    t-test and Mann-Whitney U, Holm-corrected across the tests
  * latency CDFs per scenario, a burst time series, and a failure timeline
  * a per-run CSV for reproducibility

Why run-level: the five repetitions are the independent units. Pooling requests
across runs would treat thousands of correlated samples as independent and make
every difference significant, which is the mistake the plan warns about.

Usage:
  python bench/analyse_matrix.py
  python bench/analyse_matrix.py --results scratch/matrix/results.jsonl --no-charts
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

SCENARIOS = ["steady", "ramp", "burst", "degraded", "failure"]
ALGORITHMS = ["round_robin", "weighted_rr", "least_conn", "least_response_time", "ml"]
LABELS = {
    "round_robin": "Round Robin",
    "weighted_rr": "Weighted RR",
    "least_conn": "Least Conn",
    "least_response_time": "Least RT",
    "ml": "ML (herd)",
}
COLORS = {
    "round_robin": "#8c8c8c",
    "weighted_rr": "#b08a5a",
    "least_conn": "#3f7fbf",
    "least_response_time": "#4aa06a",
    "ml": "#c0392b",
}
SCENARIO_TITLES = {
    "steady": "Steady",
    "ramp": "Ramp",
    "burst": "Burst",
    "degraded": "Mixed cost + degraded server",
    "failure": "Server failure mid-run",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cfcfcf",
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], resamples: int = 5000, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean. n is 5, so this is wide by design."""
    if len(values) < 2:
        return (values[0] if values else float("nan"),) * 2
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    result = stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def mannwhitney(a: list[float], b: list[float]) -> tuple[float, float]:
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def pct_change(a: float, b: float) -> float:
    return (a - b) / b * 100.0 if b else float("nan")


def cpu_imbalance(path: Path, cache: dict[str, float | None]) -> float | None:
    """
    Standard deviation of per-server mean CPU, in percentage points.

    The log records the CPU of the server each request was routed to, not all
    four servers at every instant. Averaging each server's own rows and taking
    the standard deviation across servers is the closest honest proxy for the
    plan's "std dev of CPU% across servers"; it is time-weighted by traffic
    rather than sampled on a fixed clock, and the results document says so.
    """
    if path.name in cache:
        return cache[path.name]
    value: float | None = None
    if path.exists():
        frame = pd.read_parquet(path, columns=["server_id", "srv_cpu_pct"])
        means = frame.dropna(subset=["srv_cpu_pct"]).groupby("server_id")["srv_cpu_pct"].mean()
        if len(means) >= 2:
            value = float(means.std(ddof=0))
    cache[path.name] = value
    return value


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("rows", 0) > 0:
            # A retried cell appends a new result under the same run_id; the
            # latest successful attempt is the one that counts.
            rows[row["run_id"]] = row
    return list(rows.values())


def cell(rows: list[dict], scenario: str, algorithm: str) -> list[dict]:
    return [r for r in rows if r["scenario"] == scenario and r["algorithm"] == algorithm]


def series(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None]


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    return statistics.mean(values), *bootstrap_ci(values)


def fmt_ms(value: float) -> str:
    return f"{value:,.0f}"


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{fmt_ms(mean)} [{fmt_ms(lo)}-{fmt_ms(hi)}]"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def summary_tables(rows: list[dict], raw: Path) -> str:
    out: list[str] = []
    out.append("### Latency by scenario (p95 ms, mean of 5 runs with bootstrap 95% CI)\n")
    header = "| Scenario | " + " | ".join(LABELS[a] for a in ALGORITHMS) + " |"
    out.append(header)
    out.append("|---|" + "---|" * len(ALGORITHMS))
    for scenario in SCENARIOS:
        cells = []
        for algorithm in ALGORITHMS:
            values = series(cell(rows, scenario, algorithm), "p95")
            cells.append(fmt_ci(*mean_ci(values)) if values else "-")
        out.append(f"| {SCENARIO_TITLES[scenario]} | " + " | ".join(cells) + " |")

    out.append("\n### Throughput (requests/s, mean of 5 runs)\n")
    out.append(header)
    out.append("|---|" + "---|" * len(ALGORITHMS))
    for scenario in SCENARIOS:
        cells = []
        for algorithm in ALGORITHMS:
            values = series(cell(rows, scenario, algorithm), "req_s")
            cells.append(f"{statistics.mean(values):,.0f}" if values else "-")
        out.append(f"| {SCENARIO_TITLES[scenario]} | " + " | ".join(cells) + " |")

    out.append("\n### Errors, SLA and fairness (mean of 5 runs)\n")
    out.append("| Scenario | Algorithm | errors % | SLA >500 ms % | Jain | p50 ms | p99 ms |")
    out.append("|---|---|---|---|---|---|---|")
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            group = cell(rows, scenario, algorithm)
            if not group:
                continue
            out.append(
                f"| {SCENARIO_TITLES[scenario]} | {LABELS[algorithm]} | "
                f"{statistics.mean(series(group, 'error_pct')):.2f} | "
                f"{statistics.mean(series(group, 'sla_violation_pct')):.1f} | "
                f"{statistics.mean(series(group, 'jain')):.3f} | "
                f"{statistics.mean(series(group, 'p50')):.0f} | "
                f"{statistics.mean(series(group, 'p99')):.0f} |"
            )

    out.append("\n### Routing overhead and CPU (mean over all runs)\n")
    out.append("| Algorithm | decision mean ms | decision max ms | balancer CPU % | generator CPU % |")
    out.append("|---|---|---|---|---|")
    for algorithm in ALGORITHMS:
        group = [r for r in rows if r["algorithm"] == algorithm]
        decisions = [r["ml"]["decision_ms_mean"] for r in group
                     if r.get("ml") and r["ml"].get("decision_ms_mean") is not None]
        decision_max = [r["ml"]["decision_ms_max"] for r in group
                        if r.get("ml") and r["ml"].get("decision_ms_max") is not None]
        balancer, generator = [], []
        for row in group:
            for name, values in (row.get("cpu") or {}).items():
                if "balancer" in name and values.get("median_pct") is not None:
                    balancer.append(values["median_pct"])
                elif "matrix_gen" in name and values.get("median_pct") is not None:
                    generator.append(values["median_pct"])
        decision_cell = f"{statistics.mean(decisions):.2f}" if decisions else "-"
        max_cell = f"{max(decision_max):.2f}" if decision_max else "-"
        cpu_cell = f"{statistics.mean(balancer):.0f}" if balancer else "-"
        gen_cell = f"{statistics.mean(generator):.0f}" if generator else "-"
        out.append(f"| {LABELS[algorithm]} | {decision_cell} | {max_cell} | {cpu_cell} | {gen_cell} |")

    out.append("\n### CPU imbalance (std dev of per-server mean CPU, percentage points)\n")
    out.append("| Scenario | " + " | ".join(LABELS[a] for a in ALGORITHMS) + " |")
    out.append("|---|" + "---|" * len(ALGORITHMS))
    cache: dict[str, float | None] = {}
    for scenario in SCENARIOS:
        cells = []
        for algorithm in ALGORITHMS:
            values = [
                cpu_imbalance(raw / r["file"], cache)
                for r in cell(rows, scenario, algorithm)
            ]
            values = [v for v in values if v is not None]
            cells.append(f"{statistics.mean(values):.1f}" if values else "-")
        out.append(f"| {SCENARIO_TITLES[scenario]} | " + " | ".join(cells) + " |")

    out.append("\n### Routing share (mean percent of requests per server)\n")
    servers = ["backend1", "backend2", "backend3", "backend4"]
    out.append("| Scenario | Algorithm | " + " | ".join(servers) + " |")
    out.append("|---|---|" + "---|" * len(servers))
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            group = cell(rows, scenario, algorithm)
            if not group:
                continue
            means = []
            for server in servers:
                values = [r.get("share", {}).get(server, 0.0) for r in group]
                means.append(f"{statistics.mean(values):.1f}")
            out.append(f"| {SCENARIO_TITLES[scenario]} | {LABELS[algorithm]} | "
                       + " | ".join(means) + " |")

    out.append("\n### Model state during ML cells (mean over the 5 runs)\n")
    out.append("| Scenario | live MAE(log) | decisions | fallbacks | drift-routed |")
    out.append("|---|---|---|---|---|")
    for scenario in SCENARIOS:
        group = [r for r in cell(rows, scenario, "ml") if r.get("ml")]
        if not group:
            continue
        maes = [r["ml"]["drift"]["live_mae_log"] for r in group
                if (r["ml"].get("drift") or {}).get("live_mae_log") is not None]
        out.append(
            f"| {SCENARIO_TITLES[scenario]} | "
            f"{statistics.mean(maes):.3f} | "
            f"{statistics.mean([r['ml'].get('decisions', 0) for r in group]):.0f} | "
            f"{sum(r['ml'].get('fallbacks', 0) for r in group)} | "
            f"{sum(r['ml'].get('drift_routed', 0) for r in group)} |"
        )

    return "\n".join(out) + "\n"


def hypothesis_results(rows: list[dict]) -> tuple[str, dict]:
    """H1-H4 from the plan's Section 11, with Holm correction."""
    out: list[str] = []
    tests: list[dict] = []

    for scenario in SCENARIOS:
        ml = series(cell(rows, scenario, "ml"), "p95")
        if not ml:
            continue
        for algorithm in ALGORITHMS:
            if algorithm == "ml":
                continue
            other = series(cell(rows, scenario, algorithm), "p95")
            if not other:
                continue
            t_stat, t_p = welch(ml, other)
            u_stat, u_p = mannwhitney(ml, other)
            tests.append({
                "scenario": scenario,
                "baseline": algorithm,
                "ml_mean": statistics.mean(ml),
                "base_mean": statistics.mean(other),
                "change_pct": pct_change(statistics.mean(ml), statistics.mean(other)),
                "welch_t": t_stat,
                "welch_p": t_p,
                "mw_u": u_stat,
                "mw_p": u_p,
            })

    adjusted = holm([t["welch_p"] for t in tests])
    adjusted_mw = holm([t["mw_p"] for t in tests])
    for test, adj, adj_mw in zip(tests, adjusted, adjusted_mw):
        test["welch_p_holm"] = adj
        test["mw_p_holm"] = adj_mw

    # H1: lower p95 than the best traditional baseline, in at least 3 of 5
    # scenarios, with a significant Holm-adjusted test against that baseline.
    wins = 0
    out.append("### H1 - p95 against the best traditional baseline per scenario\n")
    out.append("| Scenario | Best baseline | ML p95 | Baseline p95 | Change | Welch p (Holm) | MW p (Holm) | Signif. |")
    out.append("|---|---|---|---|---|---|---|---|")
    for scenario in SCENARIOS:
        ml = series(cell(rows, scenario, "ml"), "p95")
        baselines = {
            a: series(cell(rows, scenario, a), "p95")
            for a in ALGORITHMS if a != "ml"
        }
        baselines = {a: v for a, v in baselines.items() if v}
        if not ml or not baselines:
            continue
        best = min(baselines, key=lambda a: statistics.mean(baselines[a]))
        test = next(
            (t for t in tests if t["scenario"] == scenario and t["baseline"] == best), None
        )
        if test is None:
            continue
        significant = (
            test["welch_p_holm"] < 0.05 or test["mw_p_holm"] < 0.05
        ) and test["change_pct"] < 0
        wins += int(significant)
        out.append(
            f"| {SCENARIO_TITLES[scenario]} | {LABELS[best]} | "
            f"{statistics.mean(ml):.0f} | {statistics.mean(baselines[best]):.0f} | "
            f"{test['change_pct']:+.1f}% | {test['welch_p_holm']:.4f} | "
            f"{test['mw_p_holm']:.4f} | {'win' if significant else 'no'} |"
        )
    out.append(
        f"\n**H1: {'met' if wins >= 3 else 'not met'}** - ML is significantly better on "
        f"{wins} of 5 scenarios (needs 3).\n"
    )

    # H2: steady, not more than 5% worse than round robin.
    ml = series(cell(rows, "steady", "ml"), "p95")
    rr = series(cell(rows, "steady", "round_robin"), "p95")
    if ml and rr:
        ratio = statistics.mean(ml) / statistics.mean(rr)
        out.append(
            f"**H2: {'met' if ratio <= 1.05 else 'not met'}** - steady ML p95 is "
            f"{ratio:.2f}x Round Robin (needs <= 1.05x).\n"
        )

    # H3: routing overhead under 5 ms.
    ml_rows = [r for r in rows if r["algorithm"] == "ml" and r.get("ml")]
    decision_means = [r["ml"]["decision_ms_mean"] for r in ml_rows
                      if r["ml"].get("decision_ms_mean") is not None]
    decision_max = [r["ml"]["decision_ms_max"] for r in ml_rows
                    if r["ml"].get("decision_ms_max") is not None]
    if decision_means:
        out.append(
            f"**H3: {'met' if statistics.mean(decision_means) < 5.0 else 'not met'}** - "
            f"ML decision cost mean {statistics.mean(decision_means):.2f} ms, worst single "
            f"decision {max(decision_max):.2f} ms (needs mean < 5 ms).\n"
        )

    # H4: failure redirected within 5 s, with no user-visible errors.
    fault_rows = [r for r in rows if r["scenario"] == "failure" and r.get("fault")]
    hit = [r["fault"].get("last_hit_after_stop_s") for r in fault_rows
           if r["fault"].get("last_hit_after_stop_s") is not None]
    errors = series(fault_rows, "error_pct")
    back = [r["fault"].get("traffic_back_after_start_s") for r in fault_rows
            if r["fault"].get("traffic_back_after_start_s") is not None]
    if hit:
        out.append(
            f"**H4: {'met' if max(hit) < 5.0 else 'not met'}** - traffic left the stopped "
            f"server within {max(hit):.2f} s (worst of {len(hit)} runs, needs < 5 s); "
            f"mean errors {statistics.mean(errors):.2f}%; traffic returned "
            f"{statistics.mean(back):.1f} s after restart on average "
            f"(slow start included).\n"
        )

    out.append("\n### Welch and Mann-Whitney tests, all scenario/baseline pairs\n")
    out.append("| Scenario | Baseline | Change | Welch p | Welch Holm | MW p | MW Holm |")
    out.append("|---|---|---|---|---|---|---|")
    for test in sorted(tests, key=lambda t: (SCENARIOS.index(t["scenario"]), t["baseline"])):
        out.append(
            f"| {SCENARIO_TITLES[test['scenario']]} | {LABELS[test['baseline']]} | "
            f"{test['change_pct']:+.1f}% | {test['welch_p']:.4f} | "
            f"{test['welch_p_holm']:.4f} | {test['mw_p']:.4f} | "
            f"{test['mw_p_holm']:.4f} |"
        )

    return "\n".join(out) + "\n", {
        "tests": tests,
        "h1_wins": wins,
        "h2_ratio": (statistics.mean(ml) / statistics.mean(rr)) if ml and rr else None,
        "decision_ms_mean": statistics.mean(decision_means) if decision_means else None,
        "decision_ms_max": max(decision_max) if decision_max else None,
        "failure_max_leave_s": max(hit) if hit else None,
        "failure_train_back_s": statistics.mean(back) if back else None,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_p95(rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 3.6))
    width = 0.16
    positions = range(len(SCENARIOS))
    for index, algorithm in enumerate(ALGORITHMS):
        means, lower, upper = [], [], []
        for scenario in SCENARIOS:
            values = series(cell(rows, scenario, algorithm), "p95")
            if not values:
                means.append(0)
                lower.append(0)
                upper.append(0)
                continue
            mean, lo, hi = mean_ci(values)
            means.append(mean)
            lower.append(max(0.0, mean - lo))
            upper.append(max(0.0, hi - mean))
        offset = (index - (len(ALGORITHMS) - 1) / 2) * width
        axis.bar(
            [p + offset for p in positions], means, width,
            yerr=[lower, upper], capsize=2, label=LABELS[algorithm],
            color=COLORS[algorithm], edgecolor="white", linewidth=0.6,
            error_kw={"elinewidth": 0.8, "ecolor": "#555"},
        )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(
        [textwrap.fill(SCENARIO_TITLES[s], 16) for s in SCENARIOS],
        fontsize=8,
    )
    axis.set_ylabel("p95 latency (ms)")
    axis.set_title("p95 by scenario, mean of 5 runs (error bars: bootstrap 95% CI)")
    axis.legend(ncol=5, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def chart_cdfs(rows: list[dict], raw: Path, out_dir: Path) -> None:
    for scenario in SCENARIOS:
        figure, axis = plt.subplots(figsize=(5.4, 3.8))
        plotted = False
        for algorithm in ALGORITHMS:
            values: list[float] = []
            for row in cell(rows, scenario, algorithm):
                path = raw / row["file"]
                if not path.exists():
                    continue
                frame = pd.read_parquet(path, columns=["status", "response_ms"])
                ok = frame.loc[frame["status"] < 400, "response_ms"]
                # 4,000 samples per run is plenty for a CDF and keeps reading fast.
                values.extend(ok.sample(min(len(ok), 4000), random_state=1).tolist())
            if not values:
                continue
            values.sort()
            step = max(1, len(values) // 3000)
            xs = values[::step]
            ys = [(i + 1) / len(values) * 100 for i in range(0, len(values), step)]
            axis.plot(xs, ys, label=LABELS[algorithm], color=COLORS[algorithm], linewidth=1.4)
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        axis.set_xscale("log")
        axis.set_xlim(5, 5000)
        axis.set_ylim(0, 100)
        axis.set_xlabel("Latency (ms, log scale)")
        axis.set_ylabel("Percent of successful requests")
        axis.set_title(f"Latency CDF - {SCENARIO_TITLES[scenario]}")
        axis.legend(fontsize=7, loc="lower right")
        figure.tight_layout()
        figure.savefig(out_dir / f"cdf_{scenario}.png", dpi=150)
        plt.close(figure)


def chart_burst(rows: list[dict], raw: Path, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    bucket_s = 2.0
    for algorithm in ("round_robin", "least_conn", "ml"):
        buckets: dict[int, list[float]] = {}
        for row in cell(rows, "burst", algorithm):
            file = raw / row["file"]
            if not file.exists():
                continue
            frame = pd.read_parquet(file, columns=["timestamp", "status", "response_ms"])
            frame = frame[frame["status"] < 400]
            start = frame["timestamp"].min()
            offsets = ((frame["timestamp"] - start) // bucket_s * bucket_s).astype(int)
            for bucket, group in frame.groupby(offsets):
                buckets.setdefault(int(bucket), []).extend(group["response_ms"].tolist())
        points = sorted(
            (bucket, statistics.quantiles(values, n=20)[18])
            for bucket, values in buckets.items()
            if len(values) >= 20
        )
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points],
                      label=LABELS[algorithm], color=COLORS[algorithm], linewidth=1.5)
    axis.set_xlabel("Seconds into the run")
    axis.set_ylabel("p95 latency per 2 s bucket (ms)")
    axis.set_title("Burst: p95 over time, 5 runs pooled")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def chart_burst_cpu(rows: list[dict], raw: Path, path: Path) -> None:
    """Per-server CPU through the burst, from the snapshot attached to each routed request."""
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    bucket_s = 2.0
    colors = {
        "backend1": "#3f7fbf",
        "backend2": "#4aa06a",
        "backend3": "#c0392b",
        "backend4": "#b08a5a",
    }
    buckets: dict[int, dict[str, list[float]]] = {}
    for row in cell(rows, "burst", "ml"):
        file = raw / row["file"]
        if not file.exists():
            continue
        frame = pd.read_parquet(
            file, columns=["timestamp", "server_id", "srv_cpu_pct"]
        ).dropna()
        start = frame["timestamp"].min()
        frame["bucket"] = ((frame["timestamp"] - start) // bucket_s * bucket_s).astype(int)
        for (bucket, server), group in frame.groupby(["bucket", "server_id"]):
            buckets.setdefault(int(bucket), {}).setdefault(server, []).extend(
                group["srv_cpu_pct"].tolist()
            )
    for server, color in colors.items():
        points = sorted(
            (bucket, statistics.mean(values[server]))
            for bucket, values in buckets.items()
            if len(values.get(server, [])) >= 10
        )
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points],
                      label=server, color=color, linewidth=1.4)
    axis.set_xlabel("Seconds into the run")
    axis.set_ylabel("CPU (% of quota)")
    axis.set_title("Burst: per-server CPU over time, ML runs pooled")
    axis.legend(fontsize=8, ncol=4)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def chart_failure(rows: list[dict], raw: Path, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    bucket_s = 1.0
    points: dict[int, dict[str, int]] = {}
    runs = 0
    restart_offsets = []
    for row in cell(rows, "failure", "ml"):
        file = raw / row["file"]
        if not file.exists() or not row.get("fault", {}).get("stop_at"):
            continue
        runs += 1
        stop = row["fault"]["stop_at"]
        start = row["fault"].get("start_at")
        if start:
            restart_offsets.append(start - stop)
        frame = pd.read_parquet(file, columns=["timestamp", "status", "server_id"])
        frame["rel"] = ((frame["timestamp"] - stop) // bucket_s * bucket_s).astype(int)
        for rel, group in frame.groupby("rel"):
            rel = int(rel)
            if rel < -20 or rel > 30:
                continue
            slot = points.setdefault(rel, {"total": 0, "faulted": 0, "errors": 0})
            slot["total"] += len(group)
            slot["faulted"] += int((group["server_id"] == row.get("fault_server", "backend3")).sum())
            slot["errors"] += int((group["status"] >= 500).sum())

    if not points or runs == 0:
        plt.close(figure)
        return

    rels = sorted(points)
    axis.plot(rels, [points[r]["faulted"] / runs / bucket_s for r in rels],
              label="requests/s to stopped server", color="#c0392b", linewidth=1.6)
    axis.plot(rels, [points[r]["total"] / runs / bucket_s for r in rels],
              label="requests/s to all servers", color="#3f7fbf", linewidth=1.4)
    axis.plot(rels, [points[r]["errors"] / runs / bucket_s for r in rels],
              label="5xx/s", color="#333333", linewidth=1.2, linestyle="--")
    if restart_offsets:
        axis.axvline(statistics.mean(restart_offsets), color="#4aa06a",
                     linestyle=":", linewidth=1.4, label="restart")
    axis.set_xlabel("Seconds relative to the container stop")
    axis.set_ylabel("Requests per second")
    axis.set_title("Failure timeline: traffic leaves the stopped server, no 5xx")
    axis.legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse the Phase 7 matrix")
    parser.add_argument("--results", default="scratch/matrix/results.jsonl")
    parser.add_argument("--raw", default="scratch/matrix/raw")
    parser.add_argument("--out", default="scratch/analysis")
    parser.add_argument("--images", default="docs/img")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"no results at {results_path}")

    rows = load_results(results_path)
    print(f"  loaded {len(rows)} successful runs from {results_path}")

    missing = [
        (scenario, algorithm)
        for scenario in SCENARIOS
        for algorithm in ALGORITHMS
        if len(cell(rows, scenario, algorithm)) != 5
    ]
    if missing:
        print("  WARNING: cells without exactly 5 successful runs:")
        for scenario, algorithm in missing:
            print(f"    {scenario:9s} {algorithm:20s} n={len(cell(rows, scenario, algorithm))}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = Path(args.images)
    images.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw)

    tables = summary_tables(rows, raw)
    hypotheses, summary = hypothesis_results(rows)
    report = "# Phase 7 matrix analysis\n\n" + tables + "\n" + hypotheses
    (out_dir / "tables.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        fields = [
            "run_id", "scenario", "algorithm", "repeat", "rows", "span_s", "req_s",
            "p50", "p95", "p99", "max_ms", "error_pct", "sla_violation_pct", "jain",
            "cpu_std", "decision_ms_mean", "decision_ms_max", "balancer_cpu_pct",
            "generator_cpu_pct",
        ]
        writer.writerow(fields)
        cache: dict[str, float | None] = {}
        for row in rows:
            balancer = generator = None
            for name, values in (row.get("cpu") or {}).items():
                if "balancer" in name:
                    balancer = values.get("median_pct")
                elif "matrix_gen" in name:
                    generator = values.get("median_pct")
            ml = row.get("ml") or {}
            writer.writerow([
                row["run_id"], row["scenario"], row["algorithm"], row["repeat"],
                row["rows"], row["span_s"], row["req_s"], row["p50"], row["p95"],
                row["p99"], row["max_ms"], row["error_pct"], row.get("sla_violation_pct"),
                row.get("jain"), cpu_imbalance(raw / row["file"], cache),
                ml.get("decision_ms_mean"), ml.get("decision_ms_max"),
                balancer, generator,
            ])

    if not args.no_charts:
        chart_p95(rows, images / "p95_by_scenario.png")
        chart_cdfs(rows, raw, images)
        chart_burst(rows, raw, images / "burst_p95_over_time.png")
        chart_burst_cpu(rows, raw, images / "burst_cpu_over_time.png")
        chart_failure(rows, raw, images / "failure_timeline.png")
        print(f"  charts  : {images}")

    print(tables)
    print(hypotheses)
    print(f"  tables  : {out_dir / 'tables.md'}")
    print(f"  summary : {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

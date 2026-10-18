"""Quick health view of the running stack: balancer metrics per backend."""
import httpx
import sys

BALANCER = "http://localhost:9090"


def main():
    try:
        r = httpx.get(f"{BALANCER}/stats", timeout=10)
    except Exception as exc:
        print(f"balancer unreachable: {exc}")
        sys.exit(1)

    data = r.json()
    print(f"algorithm: {data['algorithm']}   sla_ms: {data['sla_ms']}\n")
    header = (
        f"{'server':10s} {'healthy':7s} {'cpu%':>7s} {'mem%':>7s} {'infl':>5s} "
        f"{'rate':>7s} {'ewma':>8s} {'p95':>8s} {'err':>6s} {'poll':>6s}"
    )
    print(header)
    print("-" * len(header))
    for s in data["servers"]:
        m = s["metrics"] or {}
        print(
            f"{s['server_id']:10s} {str(s['healthy']):7s} "
            f"{m.get('cpu_pct', 0):7.2f} {m.get('mem_pct', 0):7.2f} "
            f"{m.get('inflight_requests', 0):5d} {m.get('req_rate_5s', 0):7.2f} "
            f"{m.get('rt_ewma_ms', 0):8.2f} {m.get('rt_p95_5s_ms', 0):8.2f} "
            f"{m.get('error_rate_5s', 0):6.3f} {m.get('poll_latency_ms', 0):6.2f}"
        )

    try:
        log = httpx.get(f"{BALANCER}/admin/logger", timeout=5).json()
        print(f"\nlogger: run={log['run_id']} rows={log['rows_written']} "
              f"buffered={log['buffered']} dropped={log['dropped']}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
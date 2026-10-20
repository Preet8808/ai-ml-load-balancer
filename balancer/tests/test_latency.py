"""
Tests for the dashboard's 60 s latency window.

The window is what "p95 over time" plots, so its two properties matter: it
ignores samples older than the window, and its percentile is a defined
nearest-rank rather than an interpolation that would quietly differ from the
A/B harness.
"""

from balancer import main


def _fill(samples):
    main.RECENT_LATENCIES.clear()
    for age_s, ms in samples:
        main.RECENT_LATENCIES.append((1000.0 - age_s, float(ms)))


def test_percentiles_are_nearest_rank_over_the_window():
    _fill([(1.0, ms) for ms in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)])

    out = main.latency_window(1000.0)

    assert out["samples"] == 10
    assert out["p50"] == 50.0
    assert out["p95"] == 100.0
    assert out["p99"] == 100.0
    assert out["window_s"] == 60.0


def test_samples_older_than_the_window_are_excluded():
    _fill([(1.0, 20.0), (59.0, 40.0), (61.0, 9999.0)])

    out = main.latency_window(1000.0)

    assert out["samples"] == 2
    assert out["p95"] == 40.0, "a stale 9999 ms sample must not set the p95"


def test_an_empty_window_reports_nulls_rather_than_zero():
    _fill([])

    out = main.latency_window(1000.0)

    assert out["samples"] == 0
    assert out["p50"] is None and out["p95"] is None and out["p99"] is None

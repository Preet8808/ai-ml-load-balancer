"""Static checks for the dashboard shell.

The dashboard is a single served file, so these assertions are intentionally
about stable user-facing elements rather than styling: the developer links must
stay visible, and the run-story controls must keep unique DOM ids for the JS
bindings. Behaviour itself is exercised against the live stack.
"""

import re
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard.html"


def _html() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


def test_dashboard_keeps_raw_stats_and_api_docs_links_visible():
    html = _html()
    assert 'href="/stats"' in html
    assert 'href="/docs"' in html


def test_run_story_elements_exist_and_ids_are_unique():
    html = _html()
    required = {
        "tiles",
        "servers",
        "story-panel",
        "outcomes",
        "share-bar",
        "share-legend",
        "timeline-list",
        "run-summary",
        "run-summary-grid",
        "fleet-alert",
        "demo-start",
        "demo-stop",
        "demo-fail",
        "demo-drift",
        "demo-reset",
    }
    ids = re.findall(r'id="([^"]+)"', html)
    assert not {item for item in ids if ids.count(item) > 1}
    assert required.issubset(set(ids))


def test_routable_and_health_probe_are_shown_as_different_things():
    html = _html()
    assert '"routable"' in html
    assert "health probes" in html
    assert "probe up · not routable" in html

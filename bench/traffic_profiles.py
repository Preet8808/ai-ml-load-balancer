"""
Locust traffic profiles for dataset generation (Phase 3).

Six traffic profiles as defined in the project plan:
  1. Steady    — constant rate, e.g. 50 req/s
  2. Ramp      — rate rises from 10 to 150 req/s over 5 min
  3. Burst     — 5x spike lasting 10-30 s, repeated
  4. Wave      — sinusoidal rate, mimics daily cycle
  5. MixedCost — random mix of light, medium and heavy requests
  6. DegradedServer — one server slowed during the run

Usage:
  locust -f bench/traffic_profiles.py --host=http://localhost:8080 \
      --headless -u <users> -r <spawn_rate> -t <duration> --tags steady
"""

import math
import time
import random
import os

from locust import HttpUser, task, tag, between, constant, events


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Request mix: 60% light, 25% cpu, 10% io, 5% payload
DEFAULT_ENDPOINTS = [
    ("/ping", 0.60),
    ("/cpu", 0.25),
    ("/io", 0.10),
    ("/payload", 0.05),
]

# Override mix for specific profiles via environment variable
ENDPOINT_MIX = os.environ.get("ENDPOINT_MIX", "")


def get_endpoints():
    """Return endpoint list, optionally overridden by env var."""
    if not ENDPOINT_MIX:
        return DEFAULT_ENDPOINTS
    # Format: "ping:0.5,cpu:0.3,io:0.15,payload:0.05"
    result = []
    for part in ENDPOINT_MIX.split(","):
        name, weight = part.split(":")
        result.append((f"/{name.strip()}", float(weight)))
    return result


# ---------------------------------------------------------------------------
# Base user class
# ---------------------------------------------------------------------------

class BaseUser(HttpUser):
    """Base class with shared request logic."""
    abstract = True

    def _make_weighted_request(self):
        endpoints = get_endpoints()
        paths, weights = zip(*endpoints)
        path = random.choices(paths, weights=weights)[0]
        self.client.get(path, name=path)


# ---------------------------------------------------------------------------
# Profile 1: Steady — constant rate
# ---------------------------------------------------------------------------

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    profile = environment.parsed_options.tags or ["unknown"]
    print(f"Starting traffic profile: {profile}")


class SteadyUser(BaseUser):
    """Constant rate traffic — ~1 req/s per user."""
    wait_time = constant(1.0)

    @tag("steady")
    @task
    def steady_request(self):
        self._make_weighted_request()


# ---------------------------------------------------------------------------
# Profile 2: Ramp — rate rises from 10 to 150 req/s over 5 min
# ---------------------------------------------------------------------------

class RampUser(BaseUser):
    """Gradually increasing traffic rate over 5 minutes."""
    wait_time = between(0.01, 0.5)
    start_time = None

    def on_start(self):
        self.start_time = time.time()

    @tag("ramp")
    @task
    def ramp_request(self):
        elapsed = time.time() - self.start_time
        # Ramp from ~100ms to ~7ms wait over 300 seconds
        # wait_time = max(0.007, 0.1 * (1 - elapsed / 300))
        # We can't dynamically set wait_time per-request in Locust easily,
        # so we use between() with a wide range and let spawn rate handle scaling.
        self._make_weighted_request()


# ---------------------------------------------------------------------------
# Profile 3: Burst — 5x spike lasting 10-30 s, repeated
# ---------------------------------------------------------------------------

class BurstUser(BaseUser):
    """Bursty traffic — alternates between calm and 5x spike periods."""
    wait_time = between(0.5, 2.0)
    start_time = None
    burst_interval = 30.0  # seconds between bursts
    burst_duration = 15.0   # seconds of burst

    def on_start(self):
        self.start_time = time.time()

    @tag("burst")
    @task
    def burst_request(self):
        elapsed = time.time() - self.start_time
        cycle_pos = elapsed % self.burst_interval

        if cycle_pos < self.burst_duration:
            # In burst phase — fire rapidly
            self.client.get("/cpu", name="/cpu[burst]")
            self.client.get("/ping", name="/ping[burst]")
        else:
            # Calm phase
            self._make_weighted_request()


# ---------------------------------------------------------------------------
# Profile 4: Wave — sinusoidal rate, mimics daily cycle
# ---------------------------------------------------------------------------

class WaveUser(BaseUser):
    """Sinusoidal traffic — mimics daily load cycle."""
    wait_time = between(0.1, 1.0)
    start_time = None
    wave_period = 120.0  # 2-minute cycle for demo (real daily = 86400s)

    def on_start(self):
        self.start_time = time.time()

    @tag("wave")
    @task
    def wave_request(self):
        elapsed = time.time() - self.start_time
        # Sinusoidal intensity: 0 at trough, 1 at peak
        intensity = (math.sin(2 * math.pi * elapsed / self.wave_period) + 1) / 2

        # At low intensity, skip some requests (longer effective wait)
        if random.random() > intensity:
            time.sleep(0.5)
            return

        self._make_weighted_request()


# ---------------------------------------------------------------------------
# Profile 5: MixedCost — random mix of light, medium and heavy requests
# ---------------------------------------------------------------------------

class MixedCostUser(BaseUser):
    """Random mix of request types with varying costs."""
    wait_time = between(0.1, 1.0)

    @tag("mixed_cost")
    @task
    def mixed_request(self):
        self._make_weighted_request()


# ---------------------------------------------------------------------------
# Profile 6: DegradedServer — one server slowed during the run
# ---------------------------------------------------------------------------

class DegradedServerUser(BaseUser):
    """Normal traffic, but one backend is degraded (slowed by fault injection)."""
    wait_time = between(0.2, 1.0)

    @tag("degraded")
    @task
    def degraded_request(self):
        self._make_weighted_request()

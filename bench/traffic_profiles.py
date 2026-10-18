"""
Locust traffic profiles for dataset generation (Phase 3).

Six profiles from the project plan. Each profile is a LoadTestShape, so the
offered load genuinely varies over time instead of relying on a constant
`wait_time`:

  1. steady       constant concurrency
  2. ramp         concurrency climbs from a low base to the target
  3. burst        periodic spikes well above the baseline
  4. wave         sinusoidal concurrency
  5. mixed_cost   constant concurrency, heavier endpoint mix
  6. degraded     constant concurrency while one backend is slowed

The active profile is selected with the TRAFFIC_PROFILE environment variable.
Earlier versions relied on Locust's `--tags` filter, which still instantiates
every user class and raises "No tasks defined" for each filtered one.

Run directly with:
  set TRAFFIC_PROFILE=ramp && set TRAFFIC_USERS=120 && set BENCH_DURATION=300
  python -m locust -f bench/traffic_profiles.py --host=http://localhost:9090 --headless
"""

import math
import os
import random

from locust import HttpUser, LoadTestShape, between, events, task

TRAFFIC_PROFILE = os.environ.get("TRAFFIC_PROFILE", "steady").lower()
TRAFFIC_USERS = int(os.environ.get("TRAFFIC_USERS", "80"))
BENCH_DURATION = int(os.environ.get("BENCH_DURATION", "120"))
SPAWN_RATE = int(os.environ.get("SPAWN_RATE", "20"))

# Endpoint mix. `mixed_cost` weights the expensive endpoints far more heavily.
DEFAULT_MIX = [("/ping", 0.60), ("/cpu", 0.25), ("/io", 0.10), ("/payload", 0.05)]
MIXED_COST_MIX = [("/ping", 0.20), ("/cpu", 0.40), ("/io", 0.25), ("/payload", 0.15)]


def endpoint_mix() -> list[tuple[str, float]]:
    override = os.environ.get("ENDPOINT_MIX", "").strip()
    if override:
        parsed = []
        for part in override.split(","):
            name, weight = part.split(":")
            parsed.append((f"/{name.strip()}", float(weight)))
        return parsed
    if TRAFFIC_PROFILE == "mixed_cost":
        return MIXED_COST_MIX
    return DEFAULT_MIX


class ApiUser(HttpUser):
    """Single user class driven by whichever LoadTestShape is active."""

    wait_time = between(0.0, 0.2)

    @task
    def call(self):
        mix = endpoint_mix()
        paths, weights = zip(*mix)
        path = random.choices(paths, weights=weights)[0]

        # POST a real body for /payload so payload_kb varies and is known at
        # routing time, instead of being a constant zero for GET-only traffic.
        if path == "/payload":
            size_kb = random.choice((2, 8, 32, 128))
            self.client.post(path, data={"blob": "x" * (size_kb * 1024)}, name=path)
        else:
            self.client.get(path, name=path)


class BaseShape(LoadTestShape):
    PROFILE = "base"
    use_common_options = True  # honour -t as a hard cap
    # Locust instantiates every LoadTestShape subclass it finds, so the base
    # has to be marked abstract or startup raises on the missing tick().
    abstract = True

    def expired(self) -> bool:
        return self.get_run_time() > BENCH_DURATION


class SteadyShape(BaseShape):
    """Constant concurrency."""

    PROFILE = "steady"

    def tick(self):
        if self.expired():
            return None
        return (TRAFFIC_USERS, SPAWN_RATE)


class RampShape(BaseShape):
    """Concurrency climbs linearly from 15% to the target over the run."""

    PROFILE = "ramp"

    def tick(self):
        if self.expired():
            return None
        progress = self.get_run_time() / max(BENCH_DURATION, 1)
        base = max(2, int(TRAFFIC_USERS * 0.15))
        users = int(base + (TRAFFIC_USERS - base) * progress)
        return (users, SPAWN_RATE)


class BurstShape(BaseShape):
    """Baseline concurrency with periodic 2.5x spikes."""

    PROFILE = "burst"
    INTERVAL_S = 45.0
    BURST_S = 15.0

    def tick(self):
        if self.expired():
            return None
        phase = self.get_run_time() % self.INTERVAL_S
        if phase < self.BURST_S:
            return (int(TRAFFIC_USERS * 2.5), SPAWN_RATE)
        return (int(TRAFFIC_USERS * 0.6), SPAWN_RATE)


class WaveShape(BaseShape):
    """Sinusoidal concurrency with a 120 s period."""

    PROFILE = "wave"
    PERIOD_S = 120.0

    def tick(self):
        if self.expired():
            return None
        elapsed = self.get_run_time()
        intensity = (math.sin(2 * math.pi * elapsed / self.PERIOD_S) + 1) / 2
        users = max(2, int(TRAFFIC_USERS * (0.2 + 0.8 * intensity)))
        return (users, SPAWN_RATE)


class MixedCostShape(BaseShape):
    """Constant concurrency with the heavier endpoint mix."""

    PROFILE = "mixed_cost"

    def tick(self):
        if self.expired():
            return None
        return (TRAFFIC_USERS, SPAWN_RATE)


class DegradedShape(BaseShape):
    """Constant concurrency while the driver slows one backend."""

    PROFILE = "degraded"

    def tick(self):
        if self.expired():
            return None
        return (TRAFFIC_USERS, SPAWN_RATE)


SHAPES = {
    "steady": SteadyShape,
    "ramp": RampShape,
    "burst": BurstShape,
    "wave": WaveShape,
    "mixed_cost": MixedCostShape,
    "degraded": DegradedShape,
}


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    shape = SHAPES.get(TRAFFIC_PROFILE)
    print(
        f"profile={TRAFFIC_PROFILE} users={TRAFFIC_USERS} "
        f"duration={BENCH_DURATION}s shape={shape.__name__ if shape else 'none'}"
    )
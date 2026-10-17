#!/bin/bash
# make_dataset.sh — Full dataset generation script (Phase 3)
#
# Runs the complete data collection pipeline:
#   1. Starts the Docker Compose stack
#   2. Runs all traffic profiles with mixed routing policies
#   3. Runs probe mode for paired labels
#   4. Runs fault injection scenarios
#   5. Collects all Parquet logs
#
# Usage:
#   bash make_dataset.sh [options]
#
# Options:
#   --runs <n>        Number of runs per profile (default: 20)
#   --duration <s>    Duration per run in seconds (default: 300)
#   --output <dir>    Output directory for dataset (default: data/)
#   --profile <name>  Run only one profile (steady|ramp|burst|wave|mixed_cost|degraded)
#   --probe           Also run probe mode
#   --seed <n>        Random seed (default: 42)
#   --skip-build      Skip docker compose build
#   --help            Show this help

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUNS=20
DURATION=300
OUTPUT_DIR="data"
PROFILE=""
RUN_PROBE=false
SEED=42
SKIP_BUILD=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)       RUNS="$2"; shift 2 ;;
        --duration)   DURATION="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        --profile)     PROFILE="$2"; shift 2 ;;
        --probe)       RUN_PROBE=true; shift ;;
        --seed)        SEED="$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --help)        head -20 "$0"; exit 0 ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

export RANDOM_SEED=$SEED
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  AI/ML Load Balancer — Dataset Generation"
echo "  Runs: $RUNS  Duration: ${DURATION}s  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 0: Build and start Docker Compose stack
# ---------------------------------------------------------------------------
echo ""
echo "[step 0] Starting Docker Compose stack..."

cd "$PROJECT_ROOT"

if [ "$SKIP_BUILD" = false ]; then
    docker compose build
fi

docker compose up -d

echo "[step 0] Waiting for backends to be healthy..."
sleep 10

# Verify backends are up
for i in backend1 backend2 backend3 backend4; do
    if curl -sf "http://localhost:8080/health" >/dev/null 2>&1; then
        echo "  ✓ $i reachable"
    else
        echo "  ✗ $i unreachable (may still be starting)"
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Define traffic profiles and routing policies
# ---------------------------------------------------------------------------
PROFILES=("steady" "ramp" "burst" "wave" "mixed_cost" "degraded")
POLICIES=("random" "round_robin" "least_conn")

# Run counter for naming
RUN_ID=0

# ---------------------------------------------------------------------------
# Step 2: Run traffic collection
# ---------------------------------------------------------------------------
echo ""
echo "[step 1] Starting traffic collection..."

run_traffic() {
    local profile=$1
    local policy=$2
    local run_num=$3
    local run_id="r$(printf '%02d' $run_num)"
    local users=$4
    local spawn_rate=$5

    echo "  → Profile: $profile, Policy: $policy, Run: $run_id (users=$users)"

    # Set the routing algorithm
    curl -sf -X POST "http://localhost:8080/admin/algorithm?name=$policy" >/dev/null

    # Run Locust
    locust -f bench/traffic_profiles.py \
        --host=http://localhost:8080 \
        --headless \
        -u $users \
        -r $spawn_rate \
        -t ${DURATION}s \
        --tags "$profile" \
        --csv "$OUTPUT_DIR/${run_id}_${profile}_${policy}" \
        2>&1 | tail -5

    RUN_ID=$((RUN_ID + 1))
}

for profile in "${PROFILES[@]}"; do
    # Skip if only running one profile
    if [ -n "$PROFILE" ] && [ "$profile" != "$PROFILE" ]; then
        continue
    fi

    for policy in "${POLICIES[@]}"; do
        for run in $(seq 1 $RUNS); do
            # Adjust user count per profile
            case $profile in
                steady)      users=50;  rate=10 ;;
                ramp)        users=150; rate=5  ;;
                burst)       users=250; rate=20 ;;
                wave)        users=100; rate=10 ;;
                mixed_cost)  users=100; rate=10 ;;
                degraded)    users=80;  rate=8  ;;
                *)            users=50;  rate=10 ;;
            esac

            run_traffic "$profile" "$policy" "$run" "$users" "$rate"

            # For degraded profile, inject fault on backend3 mid-run
            if [ "$profile" = "degraded" ] && [ "$run" -le 5 ]; then
                bash bench/inject_fault.sh stress backend3 60 80 &
            fi
        done
    done
done

# ---------------------------------------------------------------------------
# Step 3: Run probe mode for paired labels
# ---------------------------------------------------------------------------
if [ "$RUN_PROBE" = true ]; then
    echo ""
    echo "[step 2] Running probe mode for paired labels..."

    python -c "
import asyncio
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from bench.probe_mode import ProbeRunner

servers = [
    {'server_id': 'backend1', 'host': 'localhost', 'port': 8001, 'cpu_limit': 2.0},
    {'server_id': 'backend2', 'host': 'localhost', 'port': 8002, 'cpu_limit': 1.0},
    {'server_id': 'backend3', 'host': 'localhost', 'port': 8003, 'cpu_limit': 1.0},
    {'server_id': 'backend4', 'host': 'localhost', 'port': 8004, 'cpu_limit': 0.5},
]

async def main():
    runner = ProbeRunner(servers, output_dir='$OUTPUT_DIR')
    await runner.run_probe(rounds=200, interval=0.5)
    stats = runner.get_ranking_stats()
    print('Probe stats:', stats)
    await runner.close()

asyncio.run(main())
"
fi

# ---------------------------------------------------------------------------
# Step 4: Collect logs from balancer
# ---------------------------------------------------------------------------
echo ""
echo "[step 3] Collecting Parquet logs..."

# The balancer writes logs to logs/ directory inside the container
# We need to copy them out
docker cp ai_ml_load_balancer-balancer-1:/app/logs "$OUTPUT_DIR/" 2>/dev/null || true

# List what we collected
echo ""
echo "Collected files:"
find "$OUTPUT_DIR" -name "*.parquet" -exec ls -lh {} \; 2>/dev/null || echo "  (no parquet files found)"

# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Dataset generation complete!"
echo "  Total runs: $RUN_ID"
echo "  Output directory: $OUTPUT_DIR"
echo ""
TOTAL_ROWS=$(python -c "
import glob, pyarrow.parquet as pq
files = glob.glob('$OUTPUT_DIR/**/*.parquet', recursive=True)
total = 0
for f in files:
    try:
        total += pq.read_table(f).num_rows
    except:
        pass
print(total)
" 2>/dev/null || echo "0")

echo "  Total rows: $TOTAL_ROWS"
echo "  Target: 200,000+ rows from 20+ runs"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 6: Tear down
# ---------------------------------------------------------------------------
echo ""
echo "[cleanup] Stopping Docker Compose stack..."
docker compose down

echo "Done."

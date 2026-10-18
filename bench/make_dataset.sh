#!/bin/bash
# make_dataset.sh - Full dataset generation for Phase 3.
#
# This is a thin wrapper around bench/quick_dataset.py, which is the single
# source of truth for the collection pipeline (algorithm switching, run
# sealing, file collection, and the signal gate). Keeping one implementation
# avoids the drift that previously left this script targeting port 8080 and
# copying Parquet files that were still being written.
#
# Usage:
#   bash bench/make_dataset.sh --runs 20 --duration 300 --seed 42
#   bash bench/make_dataset.sh --profiles steady burst --policies round_robin --runs 3
#   bash bench/make_dataset.sh --full          # the plan's full matrix
#
# Options:
#   --runs <n>       Runs per (profile, policy) cell          (default: 1)
#   --duration <s>   Seconds of traffic per run               (default: 120)
#   --profiles ...   Subset of traffic profiles
#   --policies ...   Subset of routing policies
#   --seed <n>       Random seed                             (default: 42)
#   --output <dir>   Dataset output directory                 (default: data/)
#   --full           20 runs x 300 s across all profiles/policies
#   --no-build       Skip `docker compose build`
#   --skip-gate      Do not fail if the signal gate rejects the dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RUNS=1
DURATION=120
OUTPUT_DIR="data"
SEED=42
PROFILES=()
POLICIES=()
BUILD=true
GATE=true
FULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)     RUNS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --output)   OUTPUT_DIR="$2"; shift 2 ;;
        --profiles) shift; while [[ $# -gt 0 && "$1" != --* ]]; do PROFILES+=("$1"); shift; done ;;
        --policies) shift; while [[ $# -gt 0 && "$1" != --* ]]; do POLICIES+=("$1"); shift; done ;;
        --no-build) BUILD=false; shift ;;
        --skip-gate) GATE=false; shift ;;
        --full)     FULL=true; shift ;;
        --help)     sed -n '2,25p' "$0"; exit 0 ;;
        *)          echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if $FULL; then
    RUNS=20
    DURATION=300
fi

if $BUILD; then
    echo "[1/3] Building and starting the Docker stack..."
    docker compose up -d --build
else
    echo "[1/3] Starting the Docker stack..."
    docker compose up -d
fi

# Wait for the balancer to answer before driving traffic at it
echo "[2/3] Waiting for the balancer..."
for _ in $(seq 1 60); do
    if curl -sf "http://localhost:9090/stats" >/dev/null 2>&1; then
        echo "      balancer is up"
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:9090/stats" >/dev/null 2>&1; then
    echo "Balancer never became ready. Check: docker compose logs balancer" >&2
    exit 1
fi

ARGS=(--runs "$RUNS" --duration "$DURATION" --seed "$SEED" --output "$OUTPUT_DIR")
[[ ${#PROFILES[@]} -gt 0 ]] && ARGS+=(--profiles "${PROFILES[@]}")
[[ ${#POLICIES[@]} -gt 0 ]] && ARGS+=(--policies "${POLICIES[@]}")

echo "[3/3] Collecting data..."
set +e
python bench/quick_dataset.py "${ARGS[@]}"
STATUS=$?
set -e

if [[ $STATUS -eq 2 && $GATE == true ]]; then
    echo ""
    echo "Dataset REJECTED by the signal gate. Fix the instrumentation before training."
    exit 2
fi

echo ""
echo "Done. Dataset in $OUTPUT_DIR/"
#!/bin/bash
# Fault injection scripts for dataset generation (Phase 3).
#
# These scripts inject CPU stress and network delay into specific backend
# containers to create degraded-server and noisy-network scenarios.
#
# Requirements: stress-ng and tc netem (available inside the Docker containers
# which are based on Linux images).
#
# Usage from the project root:
#   bash bench/inject_fault.sh cpu     <container> <duration_s> <load_pct>
#   bash bench/inject_fault.sh delay   <container> <duration_s> <delay_ms>
#   bash bench/inject_fault.sh loss    <container> <duration_s> <loss_pct>
#   bash bench/inject_fault.sh stress  <container> <duration_s>      # combo
#   bash bench/inject_fault.sh stop    <container>                  # kill container
#   bash bench/inject_fault.sh restore <container>                  # restart container

set -euo pipefail

ACTION="${1:-}"
CONTAINER="${2:-}"
DURATION="${3:-10}"
VALUE="${4:-}"

usage() {
    echo "Usage: $0 <action> <container> [duration_s] [value]"
    echo ""
    echo "Actions:"
    echo "  cpu     Hog CPU on a container (value = percent 0-100)"
    echo "  delay   Add network delay (value = milliseconds)"
    echo "  loss    Add packet loss (value = percent 0-100)"
    echo "  stress  Combo: CPU hog + 50ms delay"
    echo "  stop    Stop the container"
    echo "  restore Start the container back up"
    echo ""
    echo "Examples:"
    echo "  $0 cpu   backend3 30 80"
    echo "  $0 delay backend2 60 200"
    echo "  $0 stop  backend3"
    echo "  $0 restore backend3"
    exit 1
}

if [ -z "$ACTION" ] || [ -z "$CONTAINER" ]; then
    usage
fi

case "$ACTION" in
    cpu)
        LOAD="${VALUE:-80}"
        echo "[fault] CPU stress on $CONTAINER: ${LOAD}% for ${DURATION}s"
        docker exec -d "$CONTAINER" bash -c "
            stress-ng --cpu 1 --cpu-load $LOAD --timeout ${DURATION}s
        " 2>/dev/null || echo "[fault] stress-ng not available, using fallback"
        # Fallback: burn CPU with a simple loop if stress-ng is missing
        docker exec -d "$CONTAINER" bash -c "
            if ! command -v stress-ng &>/dev/null; then
                end=\$((SECONDS + $DURATION))
                while [ \$SECONDS -lt \$end ]; do
                    echo 'scale=1000; 4*a(1)' | bc -l 2>/dev/null || true
                done
            fi
        " 2>/dev/null || true
        ;;

    delay)
        DELAY_MS="${VALUE:-100}"
        echo "[fault] Network delay on $CONTAINER: ${DELAY_MS}ms for ${DURATION}s"
        docker exec "$CONTAINER" bash -c "
            tc qdisc add dev eth0 root netem delay ${DELAY_MS}ms 2>/dev/null || true
            sleep ${DURATION}
            tc qdisc del dev eth0 root 2>/dev/null || true
        " 2>/dev/null || echo "[fault] tc netem not available in container"
        ;;

    loss)
        LOSS_PCT="${VALUE:-10}"
        echo "[fault] Packet loss on $CONTAINER: ${LOSS_PCT}% for ${DURATION}s"
        docker exec "$CONTAINER" bash -c "
            tc qdisc add dev eth0 root netem loss ${LOSS_PCT}% 2>/dev/null || true
            sleep ${DURATION}
            tc qdisc del dev eth0 root 2>/dev/null || true
        " 2>/dev/null || echo "[fault] tc netem not available in container"
        ;;

    stress)
        echo "[fault] Combo stress on $CONTAINER: CPU hog + 50ms delay for ${DURATION}s"
        # CPU hog in background
        docker exec -d "$CONTAINER" bash -c "
            stress-ng --cpu 1 --cpu-load 80 --timeout ${DURATION}s 2>/dev/null || {
                end=\$((SECONDS + $DURATION))
                while [ \$SECONDS -lt \$end ]; do true; done
            }
        " 2>/dev/null || true
        # Network delay
        docker exec "$CONTAINER" bash -c "
            tc qdisc add dev eth0 root netem delay 50ms 2>/dev/null || true
            sleep ${DURATION}
            tc qdisc del dev eth0 root 2>/dev/null || true
        " 2>/dev/null || true
        ;;

    stop)
        echo "[fault] Stopping container $CONTAINER"
        docker stop "$CONTAINER"
        ;;

    restore)
        echo "[fault] Starting container $CONTAINER"
        docker start "$CONTAINER"
        ;;

    *)
        echo "Unknown action: $ACTION"
        usage
        ;;
esac

echo "[fault] Done."

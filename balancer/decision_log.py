"""
Decision log: what was chosen, why, and how it turned out.

The request log (`balancer/logger.py`) records the 16 columns the dataset and
the model are defined on, and it must keep exactly that shape. Routing
decisions need more than those columns - the reason string, the prediction, the
metrics age at decision time, and whether a model made the call at all - so
they get their own file with their own schema, joined to the request log by
`request_id`.

The row is written once per *client* request, after the last attempt, so it
records the decision that actually served and the total time the client waited.
`retries` says whether a retry happened, which is what lets an analysis exclude
retried requests from prediction-error statistics: their latency contains a
failed first attempt.

Files follow the same run lifecycle as the request log (`POST /admin/run`
seals the previous file and opens a new one), because the same argument
applies: a file with one run_id is what run-grouped analysis needs, and rows
outside a run are dropped rather than mislabelled.
"""

import os

import pyarrow as pa

from balancer.logger import RequestLogger

DECISION_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("request_id", pa.string()),
    ("run_id", pa.string()),
    ("algorithm", pa.string()),
    ("server_id", pa.string()),
    ("reason", pa.string()),
    ("predicted_ms", pa.float64()),
    ("actual_ms", pa.float64()),
    ("status", pa.int64()),
    ("retries", pa.int64()),
    ("model_driven", pa.bool_()),
    ("inflight", pa.int64()),
    ("feed_age_ms", pa.float64()),
    ("req_type", pa.string()),
])


def make_decision_logger(run_id: str = "run_0", log_dir: str | None = None) -> RequestLogger:
    """A RequestLogger writing decision rows, one file per run."""
    return RequestLogger(
        log_dir=log_dir or os.environ.get("DECISION_LOG_DIR", "decisions"),
        run_id=run_id,
        schema=DECISION_SCHEMA,
        path_prefix="decisions_",
    )

"""
ML routing algorithm (Phases 4-5).

Predicts each candidate server's response time for the request in hand and
routes to the lowest prediction. The model is trained offline by ml/train.py
and loaded from an artifact; the feature layout is the shared contract in
ml/features.py, so training and serving cannot drift apart silently.

The estimator is imported only when this algorithm is selected, and it has a
fallback: if no artifact is present, or it cannot be loaded, routing continues
on least-connections rather than failing. A load balancer that refuses to
route because a model file is missing is worse than one routing heuristically,
and /stats reports which of the two is happening.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from balancer.algorithms import LeastConnections
from balancer.interfaces import (
    Algorithm,
    RequestContext,
    ServerChoice,
    ServerInfo,
)

# Predictions within this many milliseconds of the best are treated as ties.
# The model is not accurate to better than this, and without the tie-break a
# flat model sends every request to whichever server happens to score lowest.
TIE_MS = 5.0


def _model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "models"))


def find_artifact(model_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Locate the artifact to serve.

    MODEL_PATH wins if set. Otherwise the newest training report decides: it
    records which model won and which dataset hash it was trained on, and
    globbing alone would happily pick a second-best model off alphabetical
    order. The newest report is the one that matches the current data.
    """
    explicit = os.environ.get("MODEL_PATH")
    if explicit:
        return Path(explicit)

    directory = Path(model_dir) if model_dir else _model_dir()
    if not directory.exists():
        return None

    reports = sorted(directory.glob("training_report_*.json"))
    for report in reversed(reports):
        try:
            meta = json.loads(report.read_text()).get("meta", {})
        except (OSError, ValueError):
            continue
        best, data_hash = meta.get("best_model"), meta.get("data_hash")
        if best and data_hash:
            candidate = directory / f"{best}_{data_hash}.joblib"
            if candidate.exists():
                return candidate

    artifacts = sorted(directory.glob("*.joblib"))
    return artifacts[-1] if artifacts else None


class MLRouter(Algorithm):
    """
    Route to the server with the lowest predicted response time.

    `choose_server` builds one feature row per healthy server with the shared
    serving transform, scores them in a single batched predict call, and picks
    the minimum. Every prediction is on the log1p(ms) scale the model was
    trained on; it is converted back to milliseconds before it is reported.
    """

    name = "ml"

    def __init__(self, model_path: Optional[str] = None, model_dir: Optional[Path] = None):
        self.path: Optional[Path] = Path(model_path) if model_path else None
        self.model: Any = None
        self.metadata: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        self.decisions = 0
        self.fallbacks = 0
        self._fallback = LeastConnections()
        self._model_dir = Path(model_dir) if model_dir else _model_dir()
        self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> None:
        path = self.path or find_artifact(self._model_dir)
        if path is None:
            self.load_error = f"no model artifact in {self._model_dir}"
            return

        try:
            import joblib

            bundle = joblib.load(path)
        except Exception as exc:  # missing dependency, unreadable, corrupt
            self.load_error = f"{path}: {exc}"
            return

        model, expected = bundle, None
        if isinstance(bundle, dict):
            model = bundle.get("model")
            expected = bundle.get("feature_columns")

        # The column order is part of the artifact. Serving a model whose
        # columns arrived in a different order yields plausible-looking but
        # wrong numbers, which is exactly the failure this check is for.
        from ml.features import feature_columns

        wanted = list(feature_columns())
        if expected is not None and list(expected) != wanted:
            self.load_error = f"{path.name}: feature columns do not match ml/features.py"
            return
        if model is None:
            self.load_error = f"{path.name}: artifact contains no model"
            return

        self.model = model
        self.path = path
        self.metadata = {
            key: value for key, value in (bundle.items() if isinstance(bundle, dict) else ()) if key != "model"
        }

    def status(self) -> Dict[str, Any]:
        """Serving state, surfaced through /stats so a demo can show it."""
        return {
            "loaded": self.model is not None,
            "artifact": self.path.name if self.model is not None and self.path else None,
            "estimator": type(self.model).__name__ if self.model is not None else None,
            "load_error": self.load_error,
            "decisions": self.decisions,
            "fallbacks": self.fallbacks,
            "trained_on": self.metadata.get("data_hash"),
            "train_rows": self.metadata.get("train_rows"),
        }

    # -- routing -----------------------------------------------------------

    def choose_server(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
    ) -> ServerChoice | None:
        healthy = [s for s in servers if s.healthy]
        if not healthy:
            return None

        if self.model is None:
            return self._fall_back(request, servers, reason="no model")

        try:
            from ml.features import build_serving_matrix, to_ms

            frame = build_serving_matrix(request.req_type.value, request.payload_kb, healthy)
            predicted_ms = to_ms(self.model.predict(frame))
        except Exception as exc:
            return self._fall_back(request, servers, reason=f"predict failed: {exc}")

        best = int(predicted_ms.argmin())
        tied = [i for i in range(len(healthy)) if predicted_ms[i] <= predicted_ms[best] + TIE_MS]

        # Among servers the model cannot separate, the least loaded one wins:
        # it keeps an uninformative model from concentrating all traffic on a
        # single backend, which is the failure mode a static predictor has.
        index = min(tied, key=lambda i: healthy[i].inflight)
        server = healthy[index]
        server.sent_since_snapshot += 1
        self.decisions += 1

        return ServerChoice(
            server=server,
            reason=f"ml(pred {predicted_ms[index]:.1f}ms)",
            predicted_rt_ms=float(predicted_ms[index]),
        )

    def _fall_back(
        self,
        request: RequestContext,
        servers: List[ServerInfo],
        reason: str,
    ) -> ServerChoice | None:
        choice = self._fallback.choose_server(request, servers)
        if choice is not None:
            self.fallbacks += 1
            choice.reason = f"ml-fallback ({reason})"
        return choice

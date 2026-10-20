"""
Rolling-error drift guard.

A model that was accurate on the training campaign can still be wrong in
serving: the cluster changes, a backend is replaced, the traffic mix moves, or
the artifact on disk is simply not the one the report described. The plan asks
for a monitor that demotes a model whose live error grows, and this is it: a
rolling mean absolute error in log space - the same metric the model was
selected on in `docs/dataset.md` - compared against a threshold.

Two design choices are worth stating, because the obvious alternatives do not
work:

- **The error is measured on the server that actually served the request**,
  whether or not the model chose it. Once the guard demotes the model, routing
  falls back to least-connections; if that were the end of it, the guard would
  have no fresh evidence and could only ever be re-enabled by hand. So the
  router keeps predicting in the background and reports the prediction for the
  server that served, which is what `MLRouter.choose_server` does when
  `demoted` is set.
- **The window is cleared on demotion.** Re-promotion then needs a full window
  of new evidence and cannot happen off the tail of the same bad patch that
  caused the demotion. Promotion additionally requires the mean to fall below
  `threshold * hysteresis`, so a model oscillating around the threshold does
  not flap between routed and demoted.

Thresholds default to a value that is defensible rather than tuned: the
training report measures XGBoost at 0.59 MAE(log) and the naive baseline at
1.27, so 1.0 means "the model has become no better than predicting the mean".
The guard's state is reported in /stats under `ml.drift`, including
`live_mae_log` next to the threshold, so a demo can show both.
"""

from __future__ import annotations

import collections
import logging
import math
import os
import time
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)


class DriftGuard:
    """Rolling MAE(log1p ms) monitor with demotion and automatic re-promotion."""

    def __init__(
        self,
        window: Optional[int] = None,
        min_samples: Optional[int] = None,
        threshold: Optional[float] = None,
        hysteresis: Optional[float] = None,
    ):
        self.window = window if window is not None else int(os.environ.get("DRIFT_WINDOW", "200"))
        self.min_samples = (
            min_samples if min_samples is not None else int(os.environ.get("DRIFT_MIN_SAMPLES", "50"))
        )
        self.threshold = (
            threshold if threshold is not None else float(os.environ.get("DRIFT_MAE_LOG", "1.0"))
        )
        self.hysteresis = (
            hysteresis if hysteresis is not None else float(os.environ.get("DRIFT_HYSTERESIS", "0.8"))
        )
        self.demoted = False
        self.demotions = 0
        self.promotions = 0
        self.samples = 0
        self.ignored = 0
        self.last_event: Optional[str] = None
        self.last_event_at: Optional[float] = None
        self._errors: Deque[float] = collections.deque(maxlen=self.window)

    # -- measurement -------------------------------------------------------

    @property
    def live_mae(self) -> Optional[float]:
        """Mean absolute error over the window, on the log1p scale, or None."""
        if not self._errors:
            return None
        return sum(self._errors) / len(self._errors)

    def error_of(self, predicted_ms: float, actual_ms: float) -> float:
        """
        One observation's absolute error in the units the model is trained on.

        The router reports whole milliseconds, so `log1p` puts the pair back on
        the model's own scale; the number is then directly comparable with the
        MAE the training report quotes.
        """
        return abs(math.log1p(actual_ms) - math.log1p(predicted_ms))

    def observe(self, predicted_ms: Optional[float], actual_ms: Optional[float]) -> Optional[str]:
        """
        Feed one served request's prediction and outcome.

        Returns "demoted" or "promoted" when the state changed, otherwise None.
        Values that cannot be scored (missing, non-finite, negative) are
        counted in `ignored` rather than corrupting the window.
        """
        if predicted_ms is None or actual_ms is None:
            self.ignored += 1
            return None
        try:
            predicted, actual = float(predicted_ms), float(actual_ms)
        except (TypeError, ValueError):
            self.ignored += 1
            return None
        if not (math.isfinite(predicted) and math.isfinite(actual)):
            self.ignored += 1
            return None
        if predicted <= 0.0 or actual < 0.0:
            self.ignored += 1
            return None

        self._errors.append(self.error_of(predicted, actual))
        self.samples += 1
        mean = self.live_mae

        if self.demoted:
            if len(self._errors) >= self.window and mean is not None and mean < self.threshold * self.hysteresis:
                self.demoted = False
                self.promotions += 1
                self._mark("promoted")
                logger.warning(
                    "drift guard promoted the model: live MAE(log) %.3f over %d samples",
                    mean, len(self._errors),
                )
                return "promoted"
            return None

        if len(self._errors) >= self.min_samples and mean is not None and mean > self.threshold:
            self.demoted = True
            self.demotions += 1
            self._mark("demoted")
            # Fresh evidence only: re-promotion must not ride on the tail of
            # the same bad patch.
            self._errors.clear()
            logger.warning(
                "drift guard demoted the model: live MAE(log) %.3f above threshold %.3f "
                "over %d samples; routing on least-connections until it recovers",
                mean, self.threshold, self.min_samples,
            )
            return "demoted"

        return None

    def _mark(self, event: str) -> None:
        self.last_event = event
        self.last_event_at = time.time()

    # -- control -----------------------------------------------------------

    def configure(
        self,
        threshold: Optional[float] = None,
        window: Optional[int] = None,
        min_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Re-tune the guard at runtime.

        This exists so the demotion path can be demonstrated without shipping a
        deliberately broken model: lower the threshold and the next window of
        ordinary traffic demotes the model; raise it again and a clean window
        promotes it back.
        """
        if threshold is not None:
            self.threshold = float(threshold)
        if min_samples is not None:
            self.min_samples = int(min_samples)
        if window is not None:
            self.window = int(window)
            self._errors = collections.deque(self._errors, maxlen=self.window)
        return self.status()

    def reset(self) -> Dict[str, Any]:
        """Clear the window and re-enable the model (used by tests and demos)."""
        self._errors.clear()
        self.demoted = False
        return self.status()

    def status(self) -> Dict[str, Any]:
        mean = self.live_mae
        return {
            "demoted": self.demoted,
            "live_mae_log": round(mean, 4) if mean is not None else None,
            "threshold": self.threshold,
            "window": self.window,
            "min_samples": self.min_samples,
            "samples_in_window": len(self._errors),
            "samples_total": self.samples,
            "ignored": self.ignored,
            "demotions": self.demotions,
            "promotions": self.promotions,
            "last_event": self.last_event,
            "last_event_at": self.last_event_at,
        }

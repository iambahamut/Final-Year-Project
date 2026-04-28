"""Performance instrumentation for the gesture-recognition pipeline.

This module is fully optional — the rest of the codebase is built so that a
disabled logger is a zero-cost no-op. When enabled, it records per-frame
timings, predictions, ground-truth labels, lighting metadata, and the state of
the adaptive preprocessing path, then writes everything to a CSV for offline
analysis with ``evaluate_metrics.py``.

Why the metrics matter:
- **Processing time / FPS** show whether the pipeline can sustain real-time
  control on the target hardware; latency spikes break the feel of direct
  manipulation.
- **Confusion matrix / FP / FN** quantify how often the classifier confuses
  one gesture for another, or fires when the user is doing nothing at all.
- **Lighting tags** let us measure whether adaptive preprocessing actually
  preserves accuracy across illumination conditions.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional


_GESTURE_LABELS = ("none", "pinch", "thumbs_up", "point", "flat_palm")
_LIGHTING_LABELS = ("normal", "dim", "bright_overexposed", "monitor_only")


@dataclass
class _FrameRecord:
    frame_id: int = 0
    t_capture: float = 0.0
    t_preproc_done: float = 0.0
    t_landmarks_done: float = 0.0
    t_gesture_done: float = 0.0
    processing_time_ms: float = 0.0
    preproc_time_ms: float = 0.0
    landmarks_time_ms: float = 0.0
    gesture_time_ms: float = 0.0
    gesture_latency_ms: float = 0.0
    predicted_label: str = "none"
    ground_truth_label: str = "none"
    lighting_condition: str = "normal"
    brightness_smoothed: float = float("nan")
    clahe_applied: int = 0
    gamma_applied: int = 0
    denoise_applied: int = 0
    events_fired: str = ""
    fps_rolling: float = 0.0


_CSV_FIELDS = list(asdict(_FrameRecord()).keys())


class PerformanceLogger:
    """Per-frame timing and accuracy logger.

    Lifecycle (one cycle per frame):

        logger.start_frame()
        # capture happens here (logger uses its own clock)
        logger.mark_preproc_done()
        logger.mark_landmarks_done()
        logger.mark_gesture_done(predicted_label, events_fired=...)

    The logger flushes accumulated rows to CSV every ``flush_every`` frames
    so a long session does not sit on a giant in-memory buffer, and the file
    is closed cleanly via ``close()``.
    """

    def __init__(
        self,
        output_dir: str = "logs",
        session_label: str = "default",
        fps_window: int = 100,
        flush_every: int = 100,
    ):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            output_dir, f"metrics_{session_label}_{timestamp}.csv"
        )
        self._fps_window = max(1, int(fps_window))
        self._flush_every = max(1, int(flush_every))

        self._records: list[_FrameRecord] = []
        self._frame_durations: deque = deque(maxlen=self._fps_window)
        self._frame_counter = 0
        self._last_capture: Optional[float] = None

        # Sticky per-session state — overridden by setters at any time.
        self._ground_truth = "none"
        self._lighting = "normal"
        self._preproc_flags = {
            "brightness_smoothed": float("nan"),
            "clahe_applied": 0,
            "gamma_applied": 0,
            "denoise_applied": 0,
        }

        # Active record being built up by the mark_* calls.
        self._current: Optional[_FrameRecord] = None

        # Open CSV file, write header.
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    # ------------------------------------------------------------------
    # Setters that persist across frames
    # ------------------------------------------------------------------

    def set_ground_truth(self, label: str) -> None:
        if label in _GESTURE_LABELS:
            self._ground_truth = label

    def set_lighting_condition(self, condition: str) -> None:
        if condition in _LIGHTING_LABELS:
            self._lighting = condition

    # ------------------------------------------------------------------
    # Per-frame hooks
    # ------------------------------------------------------------------

    def start_frame(self, frame_id: Optional[int] = None) -> None:
        now = time.perf_counter()
        if self._last_capture is not None:
            self._frame_durations.append(now - self._last_capture)
        self._last_capture = now

        rec = _FrameRecord()
        rec.frame_id = frame_id if frame_id is not None else self._frame_counter
        rec.t_capture = now
        rec.ground_truth_label = self._ground_truth
        rec.lighting_condition = self._lighting
        self._current = rec
        self._frame_counter += 1

    def mark_preproc_done(self, preproc_flags: Optional[dict] = None) -> None:
        if self._current is None:
            return
        self._current.t_preproc_done = time.perf_counter()
        if preproc_flags:
            for key in self._preproc_flags:
                if key in preproc_flags:
                    self._preproc_flags[key] = preproc_flags[key]

    def mark_landmarks_done(self) -> None:
        if self._current is None:
            return
        self._current.t_landmarks_done = time.perf_counter()

    def mark_gesture_done(
        self,
        pred_label: str,
        events_fired: Optional[str] = None,
    ) -> None:
        if self._current is None:
            return
        rec = self._current
        rec.t_gesture_done = time.perf_counter()
        rec.predicted_label = pred_label if pred_label in _GESTURE_LABELS else "none"
        rec.events_fired = events_fired or ""

        rec.processing_time_ms = (rec.t_gesture_done - rec.t_capture) * 1000.0
        rec.preproc_time_ms = (rec.t_preproc_done - rec.t_capture) * 1000.0
        rec.landmarks_time_ms = (rec.t_landmarks_done - rec.t_preproc_done) * 1000.0
        rec.gesture_time_ms = (rec.t_gesture_done - rec.t_landmarks_done) * 1000.0
        # End-to-end latency: capture → events committed at end of pipeline.
        # Conservative upper bound — the actual pynput call is dispatched
        # inside process_*_hand before mark_gesture_done fires, so the real
        # latency is slightly lower.
        rec.gesture_latency_ms = rec.processing_time_ms

        rec.brightness_smoothed = float(self._preproc_flags["brightness_smoothed"])
        rec.clahe_applied = int(self._preproc_flags["clahe_applied"])
        rec.gamma_applied = int(self._preproc_flags["gamma_applied"])
        rec.denoise_applied = int(self._preproc_flags["denoise_applied"])

        rec.fps_rolling = self.get_rolling_fps()

        self._records.append(rec)
        self._current = None

        if len(self._records) >= self._flush_every:
            self.flush()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_rolling_fps(self) -> float:
        if not self._frame_durations:
            return 0.0
        mean_dt = sum(self._frame_durations) / len(self._frame_durations)
        return 1.0 / mean_dt if mean_dt > 0 else 0.0

    def get_last_processing_ms(self) -> float:
        return self._records[-1].processing_time_ms if self._records else 0.0

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def flush(self) -> None:
        if not self._records:
            return
        self._writer.writerows(asdict(r) for r in self._records)
        self._file.flush()
        self._records.clear()

    def close(self) -> None:
        self.flush()
        if not self._file.closed:
            self._file.close()


class NullLogger:
    """No-op logger — used when metrics are disabled.

    Implements the same surface so call sites don't need ``if logger:`` guards.
    """

    path = None

    def set_ground_truth(self, label): pass
    def set_lighting_condition(self, condition): pass
    def start_frame(self, frame_id=None): pass
    def mark_preproc_done(self, preproc_flags=None): pass
    def mark_landmarks_done(self): pass
    def mark_gesture_done(self, pred_label, events_fired=None): pass
    def get_rolling_fps(self): return 0.0
    def get_last_processing_ms(self): return 0.0
    def flush(self): pass
    def close(self): pass


# Convenience: stable label list for downstream analysis.
GESTURE_LABELS = _GESTURE_LABELS
LIGHTING_LABELS = _LIGHTING_LABELS

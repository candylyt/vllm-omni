# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
"""
Per-phase timing instrumentation for MOSS-TTS-Local Stage 0.

This module is **opt-in** via the ``MOSS_TTS_TIMING=1`` environment variable.
When the variable is unset (the default) every helper is a near-zero-cost
no-op — the context managers yield immediately, no CUDA events are
allocated, and no Python lists grow.

Usage
-----
    MOSS_TTS_TIMING=1 python throughput_moss_tts_local.py \\
        --model "$MOSS_TTS_LOCAL_PATH" \\
        --repo "$REPO" \\
        --batch-sizes 1 \\
        --output-dir ./timing_run

When the worker process exits (via the atexit hook registered below), a
table is printed to stdout summarising mean / p50 / p99 / count for every
instrumented phase, sorted by mean wall time.  Run the same script on the
``moss-tts-local`` baseline branch to get a comparable report and diff the
two tables to quantify the KV-cache speedup per phase.

Two flavours of timer
---------------------
* ``timer.gpu(name)`` — wraps a region whose work happens on the GPU.
  Uses ``torch.cuda.Event`` so the recorded duration is true GPU time
  (kernel launch + execution).  Events are queued on the CUDA stream
  without blocking the host; the elapsed times are computed at dump time
  via a single ``torch.cuda.synchronize()``.

* ``timer.cpu(name)`` — wraps a region whose work happens on the CPU
  (Python bookkeeping, FSM walks, list comprehensions).  Uses
  ``time.perf_counter_ns`` and adds the duration to a list.

You can mix both around the same logical phase if it has both a CPU and a
GPU component (rare — usually one dominates).

Resetting after warmup
----------------------
``MossTTSARStageModel._clear_warmup_state`` calls ``_TIMER.reset()`` so
the warmup / profiling pass that vLLM runs at engine init does not
pollute the post-warmup statistics.
"""

from __future__ import annotations

import atexit
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch

# Cheap module-level flag; checked at every helper entry.  Off by default.
_TIMING_ENABLED: bool = os.environ.get("MOSS_TTS_TIMING", "0") == "1"


class _Stage0Timing:
    """Per-phase timing aggregator for one Stage-0 worker process."""

    def __init__(self) -> None:
        # GPU phases store (start_event, end_event) pairs; durations are
        # computed in dump() to avoid forcing a CUDA sync per call.
        self._gpu_events: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        # CPU phases store durations (in milliseconds) directly.
        self._cpu_ms: dict[str, list[float]] = defaultdict(list)
        self._dumped: bool = False

    @contextmanager
    def gpu(self, name: str):
        """Time a GPU region.  No-op when MOSS_TTS_TIMING is unset."""
        if not _TIMING_ENABLED:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._gpu_events[name].append((start, end))

    @contextmanager
    def cpu(self, name: str):
        """Time a CPU region.  No-op when MOSS_TTS_TIMING is unset."""
        if not _TIMING_ENABLED:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._cpu_ms[name].append((time.perf_counter_ns() - t0) / 1e6)

    def reset(self) -> None:
        """Drop all accumulated samples (e.g. after warmup)."""
        if not _TIMING_ENABLED:
            return
        # Force-finish any queued events so we can safely drop them.
        if self._gpu_events:
            torch.cuda.synchronize()
        self._gpu_events.clear()
        self._cpu_ms.clear()
        self._dumped = False

    def dump(self) -> None:
        """Print a summary table.  Idempotent — safe to call multiple times."""
        if not _TIMING_ENABLED or self._dumped:
            return
        self._dumped = True

        # Resolve GPU events to milliseconds (one sync, then cheap reads).
        gpu_ms: dict[str, list[float]] = {}
        if self._gpu_events:
            torch.cuda.synchronize()
            for name, pairs in self._gpu_events.items():
                gpu_ms[name] = [s.elapsed_time(e) for s, e in pairs]

        if not gpu_ms and not self._cpu_ms:
            return

        rows: list[tuple[str, str, list[float]]] = []
        for name, samples in gpu_ms.items():
            rows.append((name, "GPU", samples))
        for name, samples in self._cpu_ms.items():
            if name in gpu_ms:
                # Same name has both CPU and GPU samples — show CPU separately
                rows.append((f"{name} (cpu)", "CPU", samples))
            else:
                rows.append((name, "CPU", samples))

        # Sort by mean wall time, descending — biggest cost first.
        rows.sort(key=lambda r: -(sum(r[2]) / max(len(r[2]), 1)))

        line = "═" * 86
        thin = "─" * 86
        print()
        print(line)
        print("  MOSS-TTS Stage 0 — per-phase timing")
        print(f"  pid={os.getpid()}   MOSS_TTS_TIMING={os.environ.get('MOSS_TTS_TIMING')}")
        print(line)
        print(
            f"  {'phase':<38} {'kind':<5} {'mean':>9} {'p50':>9} {'p99':>9} {'count':>8}"
        )
        print(thin)
        for name, kind, samples in rows:
            samples_sorted = sorted(samples)
            n = len(samples)
            mean = sum(samples) / n
            p50 = samples_sorted[n // 2]
            p99_idx = min(n - 1, max(0, int(n * 0.99) - 1))
            p99 = samples_sorted[p99_idx]
            print(
                f"  {name:<38} {kind:<5} {mean:>9.3f} {p50:>9.3f} {p99:>9.3f} {n:>8}"
            )
        print(thin)
        print("  All durations in milliseconds.  Sorted by mean (biggest first).")
        print(line)


# Module-level singleton.  The model imports this and wraps phases.
_TIMER = _Stage0Timing()


def get_timer() -> _Stage0Timing:
    """Return the process-wide timer.  Always safe to call."""
    return _TIMER


# Print the report at process shutdown.  Registered unconditionally so the
# user never has to remember to flush — but dump() itself is a no-op when
# timing is disabled, so this is free.
atexit.register(_TIMER.dump)

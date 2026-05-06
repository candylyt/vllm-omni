# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import threading
import time
from collections import deque
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class OmniTransferAdapterBase:
    """Base class for managing data transfer via OmniConnector.

    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """

    def __init__(self, config: Any):
        self.config = config
        if not hasattr(self, "connector"):
            self.connector = None
        # Requests that are waiting to be polled
        self._pending_load_reqs = deque()
        # Requests that have successfully retrieved data
        self._finished_load_reqs = set()
        self._cancelled_load_reqs: set[str] = set()

        # Requests that are waiting to be saved
        self._pending_save_reqs = deque()
        # Requests that have successfully saved data
        self._finished_save_reqs = set()

        self.stop_event = threading.Event()
        self._recv_cond = threading.Condition()
        self._save_cond = threading.Condition()
        connector_config = getattr(getattr(self, "connector", None), "config", {}) or {}
        self._recv_poll_wait_s = float(
            connector_config.get("connector_poll_wait_s", 0.001)
        )
        self._profile_async = _config_bool(connector_config.get("profile_async_transfer"))
        self._profile_stats = {
            "poll_attempts": 0,
            "poll_successes": 0,
            "poll_wall_s": 0.0,
            "recv_backoff_s": 0.0,
            "save_enqueued": 0,
            "save_skipped": 0,
            "save_tasks": 0,
            "save_wall_s": 0.0,
        }
        if self._profile_async:
            atexit.register(self._log_profile_stats)

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def recv_loop(self):
        """Loop to poll for incoming data.

        Process each pending request exactly once per pass.  When no request
        made progress, back off 1 ms instead of tight-spinning on failed
        shm_open syscalls (which can burn a full CPU core).
        """
        while not self.stop_event.is_set():
            n = len(self._pending_load_reqs)
            any_success = False
            for _ in range(n):
                if not self._pending_load_reqs:
                    break
                request = self._pending_load_reqs.popleft()
                request_id = request.request_id
                if request_id in self._cancelled_load_reqs:
                    self._cancelled_load_reqs.discard(request_id)
                    continue
                self.request_ids_mapping[request_id] = request.external_req_id
                try:
                    poll_t0 = time.perf_counter()
                    is_success = self._poll_single_request(request)
                    poll_dt = time.perf_counter() - poll_t0
                    if self._profile_async:
                        self._profile_stats["poll_attempts"] += 1
                        self._profile_stats["poll_wall_s"] += poll_dt
                    if is_success:
                        if self._profile_async:
                            self._profile_stats["poll_successes"] += 1
                        any_success = True
                    else:
                        self._pending_load_reqs.append(request)
                except Exception as e:
                    self._pending_load_reqs.append(request)
                    logger.warning(f"Error receiving data for {request_id}: {e}")

            # Timeout is the fallback for lock-free append/notify races.
            with self._recv_cond:
                if not self._pending_load_reqs and not self.stop_event.is_set():
                    self._recv_cond.wait(timeout=0.1)
                elif not any_success and not self.stop_event.is_set():
                    wait_s = self._recv_poll_wait_s
                    if self._profile_async:
                        self._profile_stats["recv_backoff_s"] += wait_s
                    if wait_s > 0:
                        self._recv_cond.wait(timeout=wait_s)

    def save_loop(self):
        """Loop to send outgoing data."""
        while not self.stop_event.is_set():
            while self._pending_save_reqs:
                task = self._pending_save_reqs.popleft()
                try:
                    save_t0 = time.perf_counter()
                    self._send_single_request(task)
                    if self._profile_async:
                        self._profile_stats["save_tasks"] += 1
                        self._profile_stats["save_wall_s"] += time.perf_counter() - save_t0
                except Exception as e:
                    logger.warning(f"Error saving data for {task.get('request_id')}: {e}")

            with self._save_cond:
                if not self._pending_save_reqs and not self.stop_event.is_set():
                    self._save_cond.wait(timeout=0.1)

    def _poll_single_request(self, *args, **kwargs):
        """Poll connector for a single request task.
        Subclasses should implement request-specific receive behavior."""
        raise NotImplementedError

    def _send_single_request(self, *args, **kwargs):
        """Send one pending save request task to the connector.
        Subclasses should implement task-specific handling logic."""
        raise NotImplementedError

    def load_async(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save_async(self, *args, **kwargs):
        """Submit data to be saved. To be implemented by subclasses."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Load request data from connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save data to connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        """Get finished loaded or saved requests"""
        raise NotImplementedError

    def shutdown(self):
        """Stop background loops and close the connector."""
        self.stop_event.set()
        with self._recv_cond:
            self._recv_cond.notify_all()
        with self._save_cond:
            self._save_cond.notify_all()
        if self.connector is not None:
            try:
                if self._profile_async:
                    logger.warning(
                        self._format_profile_stats(),
                    )
                self.connector.close()
            except Exception:
                pass

    def _format_profile_stats(self) -> str:
        return (
            "[AsyncTransferProfile] poll_attempts=%d poll_successes=%d "
            "poll_wall_s=%.6f recv_backoff_s=%.6f save_enqueued=%d "
            "save_skipped=%d save_tasks=%d save_wall_s=%.6f"
        ) % (
            self._profile_stats["poll_attempts"],
            self._profile_stats["poll_successes"],
            self._profile_stats["poll_wall_s"],
            self._profile_stats["recv_backoff_s"],
            self._profile_stats["save_enqueued"],
            self._profile_stats["save_skipped"],
            self._profile_stats["save_tasks"],
            self._profile_stats["save_wall_s"],
        )

    def _log_profile_stats(self) -> None:
        if self._profile_async:
            logger.warning(self._format_profile_stats())

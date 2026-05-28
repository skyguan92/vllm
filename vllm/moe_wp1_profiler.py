# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated MoE/EP profiler for A800 Kimi WP1 attribution.

The profiler is intentionally standalone and best-effort. It must not affect
normal serving unless VLLM_MOE_WP1_PROFILE=1 is set.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _safe_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return str(value)


def tensor_info(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }


def _cuda_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


class _MoEWP1Profiler:
    def __init__(self) -> None:
        self.enabled = _env_bool("VLLM_MOE_WP1_PROFILE")
        self.profile_dir = Path(
            os.environ.get("VLLM_MOE_WP1_PROFILE_DIR", "/tmp/vllm-moe-wp1-profile")
        )
        self.sample_every = _env_int("VLLM_MOE_WP1_PROFILE_SAMPLE_EVERY", 1, 1)
        self.max_records = _env_int("VLLM_MOE_WP1_PROFILE_MAX_RECORDS", 200000, 0)
        self.sync_cuda = _env_bool("VLLM_MOE_WP1_PROFILE_SYNC")
        self.dense_histograms = _env_bool("VLLM_MOE_WP1_PROFILE_DENSE_HISTOGRAM")

        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.rank = _safe_int_env("RANK")
        self.local_rank = _safe_int_env("LOCAL_RANK")
        self.world_size = _safe_int_env("WORLD_SIZE")

        self._lock = threading.Lock()
        self._record_index = 0
        self._file = None
        self._disabled_reason: str | None = None

        if self.enabled:
            atexit.register(self.close)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _open_file(self):
        if self._file is None:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            rank_tag = self.rank if self.rank is not None else "na"
            local_rank_tag = self.local_rank if self.local_rank is not None else "na"
            filename = (
                f"moe_wp1_{self.hostname}_rank{rank_tag}_local{local_rank_tag}"
                f"_pid{self.pid}.jsonl"
            )
            path = (
                self.profile_dir
                / filename
            )
            self._file = path.open("a", encoding="utf-8", buffering=1)
        return self._file

    def _should_record_locked(self) -> bool:
        if not self.enabled:
            return False
        if self.max_records and self._record_index >= self.max_records:
            return False
        self._record_index += 1
        return (self._record_index - 1) % self.sample_every == 0

    def should_record(self) -> bool:
        if _cuda_capture_active():
            return False
        with self._lock:
            return self._should_record_locked()

    def sync_if_needed(self) -> None:
        if not self.sync_cuda:
            return
        if torch.cuda.is_available() and not _cuda_capture_active():
            torch.cuda.synchronize()

    def record(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        row = {
            "ts_ns": time.time_ns(),
            "hostname": self.hostname,
            "pid": self.pid,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
        }
        row.update(event)
        try:
            payload = json.dumps(_sanitize(row), sort_keys=True, separators=(",", ":"))
            with self._lock:
                self._open_file().write(payload + "\n")
        except Exception as exc:  # pragma: no cover - best-effort guardrail.
            self._disabled_reason = repr(exc)
            self.enabled = False

    def record_histogram(
        self,
        *,
        phase: str,
        layer: str | None,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.should_record():
            return
        try:
            flat = topk_ids.detach().reshape(-1).to(dtype=torch.int64)
            valid = (flat >= 0) & (flat < global_num_experts)
            invalid_routes = int((~valid).sum().item())
            valid_flat = flat[valid]
            counts = torch.bincount(
                valid_flat, minlength=global_num_experts
            ).detach().cpu()
            nonzero = counts.nonzero(as_tuple=False).flatten()
            count_values = counts[nonzero]
            sparse_counts = [
                [int(expert), int(count)]
                for expert, count in zip(nonzero.tolist(), count_values.tolist())
            ]
            max_count = int(count_values.max().item()) if count_values.numel() else 0
            total_routes = int(valid_flat.numel())
            active_experts = int(nonzero.numel())
            mean_active = total_routes / active_experts if active_experts else 0.0
            row = {
                "type": "histogram",
                "phase": phase,
                "layer": layer,
                "top_k": top_k,
                "global_num_experts": global_num_experts,
                "tokens": int(topk_ids.shape[0]) if topk_ids.ndim > 0 else 0,
                "routes": int(flat.numel()),
                "valid_routes": total_routes,
                "invalid_routes": invalid_routes,
                "active_experts": active_experts,
                "max_expert_tokens": max_count,
                "mean_active_expert_tokens": mean_active,
                "counts_sparse": sparse_counts,
            }
            if self.dense_histograms:
                row["counts"] = [int(x) for x in counts.tolist()]
            if metadata:
                row.update(metadata)
            self.record(row)
        except Exception as exc:  # pragma: no cover - profiling must not break serving.
            self.record(
                {
                    "type": "histogram_error",
                    "phase": phase,
                    "layer": layer,
                    "error": repr(exc),
                }
            )


_PROFILER = _MoEWP1Profiler()


def enabled() -> bool:
    return _PROFILER.enabled


def record_expert_histogram(
    *,
    phase: str,
    layer: str | None,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    top_k: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    _PROFILER.record_histogram(
        phase=phase,
        layer=layer,
        topk_ids=topk_ids,
        global_num_experts=global_num_experts,
        top_k=top_k,
        metadata=metadata,
    )


@contextmanager
def profile_block(
    phase: str,
    *,
    layer: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    if not _PROFILER.should_record():
        yield
        return

    error: str | None = None
    start_ns = 0
    try:
        _PROFILER.sync_if_needed()
        start_ns = time.perf_counter_ns()
        yield
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            _PROFILER.sync_if_needed()
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            row = {
                "type": "timing",
                "phase": phase,
                "layer": layer,
                "elapsed_ms": elapsed_ms,
                "cuda_synchronized": _PROFILER.sync_cuda,
                "status": "error" if error else "ok",
            }
            if error:
                row["error"] = error
            if metadata:
                row.update(metadata)
            _PROFILER.record(row)
        except Exception:
            pass

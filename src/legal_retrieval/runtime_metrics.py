from __future__ import annotations

import math
import platform
from pathlib import Path
from typing import Any


def directory_size_bytes(path: Path) -> int:
    """Return the total size of regular files below path."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "first_query_seconds": None,
            "warm_mean_seconds": None,
        }
    total = sum(samples)
    warm = samples[1:]
    return {
        "count": len(samples),
        "total_seconds": total,
        "mean_seconds": total / len(samples),
        "median_seconds": _percentile(samples, 0.5),
        "p95_seconds": _percentile(samples, 0.95),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        # The first query may include lazy reranker/model initialization.
        "first_query_seconds": samples[0],
        "warm_mean_seconds": sum(warm) / len(warm) if warm else None,
    }


def reset_gpu_peak_memory() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
    except (ImportError, RuntimeError):
        return


def synchronize_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        return


def resource_usage() -> dict[str, Any]:
    peak_ram_bytes: int | None = None
    try:
        import resource

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        peak_ram_bytes = peak_rss if platform.system() == "Darwin" else peak_rss * 1024
    except (ImportError, OSError, ValueError):
        pass

    gpu_devices: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            synchronize_gpu()
            for device_index in range(torch.cuda.device_count()):
                gpu_devices.append({
                    "index": device_index,
                    "name": torch.cuda.get_device_name(device_index),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
                })
    except (ImportError, RuntimeError):
        pass

    return {
        "peak_process_ram_bytes": peak_ram_bytes,
        "cuda_available": bool(gpu_devices),
        "gpu_devices": gpu_devices,
    }

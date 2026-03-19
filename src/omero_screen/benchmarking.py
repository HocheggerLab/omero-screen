"""Benchmarking utilities for measuring pipeline performance.

Provides a lightweight timer that records wall-clock times for named stages,
collects system metadata, and writes a JSON report. When benchmarking is
disabled the module returns a no-op singleton with zero overhead.

Typical usage::

    from omero_screen.benchmarking import get_benchmark, init_benchmark

    init_benchmark(enabled=True, plate_id=12345)
    timer = get_benchmark()

    with timer.stage("flatfield"):
        ...

    with timer.image(image_id=1001, well="A1"):
        with timer.stage("nucleus_segmentation"):
            ...

    timer.save_report()
"""

import json
import os
import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


@dataclass
class _ImageRecord:
    """Timing record for a single image."""

    image_id: int
    well: str
    stages: dict[str, float] = field(default_factory=dict)
    start: float = 0.0
    total_s: float = 0.0


def _collect_system_info() -> dict[str, Any]:
    """Collect system metadata for the benchmark report."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }

    # RAM
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["ram_gb"] = None

    # GPU / compute device
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            info["gpu"] = "mps"
        else:
            info["gpu"] = "cpu"
    except ImportError:
        info["gpu"] = "unknown"

    # Cellpose version
    try:
        import cellpose

        info["cellpose_version"] = cellpose.__version__
    except (ImportError, AttributeError):
        info["cellpose_version"] = "unknown"

    return info


class BenchmarkTimer:
    """Active benchmark timer that records stage and per-image timings."""

    def __init__(self, plate_id: int) -> None:
        """Initialise benchmark timer for the given plate."""
        self._plate_id = plate_id
        self._stages: dict[str, float] = {}
        self._stage_starts: dict[str, float] = {}
        self._images: list[_ImageRecord] = []
        self._current_image: _ImageRecord | None = None
        self._plate_start = time.perf_counter()
        self._n_wells = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named stage. Works both at plate level and within an image context."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if self._current_image is not None:
                self._current_image.stages[name] = elapsed
            else:
                self._stages[name] = elapsed

    @contextmanager
    def image(self, image_id: int, well: str = "") -> Iterator[None]:
        """Time processing for a single image."""
        record = _ImageRecord(image_id=image_id, well=well)
        record.start = time.perf_counter()
        self._current_image = record
        try:
            yield
        finally:
            record.total_s = time.perf_counter() - record.start
            self._images.append(record)
            self._current_image = None

    def set_well_count(self, n: int) -> None:
        """Record the number of wells processed."""
        self._n_wells = n

    def save_report(self, directory: Path | None = None) -> Path:
        """Write the benchmark report as JSON and return the file path.

        Args:
            directory: Output directory. Defaults to ``<project_root>/benchmarks/``.
        """
        from omero_screen.config import project_root

        total_s = time.perf_counter() - self._plate_start
        self._stages["total_s"] = total_s

        per_image = [
            {
                "image_id": img.image_id,
                "well": img.well,
                **{f"{k}_s": round(v, 4) for k, v in img.stages.items()},
                "total_s": round(img.total_s, 4),
            }
            for img in self._images
        ]

        summary = self._compute_summary()

        report = {
            "system": _collect_system_info(),
            "plate_id": self._plate_id,
            "n_wells": self._n_wells,
            "n_images": len(self._images),
            "stages": {k: round(v, 4) for k, v in self._stages.items()},
            "per_image": per_image,
            "summary": summary,
        }

        out_dir = directory or (project_root / "benchmarks")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        path = out_dir / f"benchmark_plate_{self._plate_id}_{timestamp}.json"
        path.write_text(json.dumps(report, indent=2))
        return path

    def _compute_summary(self) -> dict[str, float | None]:
        """Compute aggregate statistics across all images."""
        if not self._images:
            return {}

        totals = [img.total_s for img in self._images]
        summary: dict[str, float | None] = {
            "total_mean_s": round(mean(totals), 4),
            "total_median_s": round(median(totals), 4),
            "total_std_s": round(stdev(totals), 4) if len(totals) > 1 else 0.0,
        }

        # Per-stage summaries
        all_stage_names: set[str] = set()
        for img in self._images:
            all_stage_names.update(img.stages.keys())

        for stage_name in sorted(all_stage_names):
            values = [
                img.stages[stage_name]
                for img in self._images
                if stage_name in img.stages
            ]
            if values:
                summary[f"{stage_name}_mean_s"] = round(mean(values), 4)
                summary[f"{stage_name}_median_s"] = round(median(values), 4)
                if len(values) > 1:
                    summary[f"{stage_name}_std_s"] = round(stdev(values), 4)

        # Throughput
        total_time = self._stages.get("total_s", 0)
        if total_time > 0:
            summary["images_per_minute"] = round(
                len(self._images) / (total_time / 60), 2
            )

        return summary


class _NoOpTimer:
    """No-op timer returned when benchmarking is disabled. Zero overhead."""

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:  # noqa: ARG002
        yield

    @contextmanager
    def image(self, image_id: int, well: str = "") -> Iterator[None]:  # noqa: ARG002
        yield

    def set_well_count(self, n: int) -> None:  # noqa: ARG002
        pass

    def save_report(self, directory: Path | None = None) -> Path:  # noqa: ARG002
        return Path()


_instance: BenchmarkTimer | _NoOpTimer = _NoOpTimer()


def init_benchmark(enabled: bool, plate_id: int = 0) -> None:
    """Initialise the benchmark singleton.

    Args:
        enabled: Whether to activate benchmarking.
        plate_id: OMERO plate ID (used in the report filename).
    """
    global _instance  # noqa: PLW0603
    _instance = BenchmarkTimer(plate_id) if enabled else _NoOpTimer()


def get_benchmark() -> BenchmarkTimer | _NoOpTimer:
    """Return the current benchmark timer (active or no-op)."""
    return _instance

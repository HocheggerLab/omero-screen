"""Rich-based progress display for the omero-screen analysis pipeline.

Provides a single live terminal panel showing plate/well/image progress,
current pipeline stage, elapsed time, and ETA — replacing the flood of
individual tqdm bars that appeared previously.

Typical usage::

    with ScreenProgress(plate_id=12345, n_wells=96) as prog:
        for well in wells:
            with prog.well(well_pos, n_images=21):
                for img in images:
                    prog.image_done()
"""

import time
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager

from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from omero_screen.config import get_console, get_logger

logger = get_logger(__name__)

# Sentinel used when no ScreenProgress instance is active
_NOOP = object()


class ScreenProgress:
    """Single live display for the entire plate analysis run.

    Shares the same Rich console as the logger (if configured).

    Args:
        plate_id: OMERO plate ID shown in the panel header.
        n_wells: Total number of non-empty wells to process.
    """

    def __init__(self, plate_id: int, n_wells: int) -> None:  # noqa: D107
        self._plate_id = plate_id
        self._n_wells = n_wells
        self._wells_done = 0
        self._current_well = ""
        self._stage = "starting"
        self._img_done = 0
        self._img_total = 0
        self._well_start: float = 0.0
        self._plate_start: float = time.monotonic()
        self._well_times: list[float] = []  # seconds per completed well

        # Two Progress objects: one for wells, one for images in current well
        self._well_progress = Progress(
            TextColumn("[bold cyan]Wells   [/]"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("  "),
            TimeElapsedColumn(),
        )
        self._img_progress = Progress(
            TextColumn("[bold green]Images  [/]"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("  [dim]{task.description}[/]"),
        )

        self._well_task: TaskID = self._well_progress.add_task(
            "wells", total=n_wells
        )
        self._img_task: TaskID = self._img_progress.add_task("", total=1)
        self._live = Live(
            self._render(), refresh_per_second=1, console=get_console()
        )

    # ------------------------------------------------------------------
    # Public context manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "ScreenProgress":  # noqa: D105
        self._live.start()
        return self

    def __exit__(self, *_: object) -> None:  # noqa: D105
        self._live.stop()

    @contextmanager
    def well(
        self, well_pos: str, n_images: int
    ) -> Generator[None, None, None]:
        """Context manager for a single well.

        Args:
            well_pos: Human-readable well position, e.g. ``"A07"``.
            n_images: Number of images to process in this well.
        """
        self._current_well = well_pos
        self._img_done = 0
        self._img_total = n_images
        self._well_start = time.monotonic()
        self._img_progress.reset(
            self._img_task,
            total=n_images,
            description=f"{well_pos}",
        )
        self._refresh()
        try:
            yield
        finally:
            elapsed = time.monotonic() - self._well_start
            self._well_times.append(elapsed)
            self._wells_done += 1
            self._well_progress.update(
                self._well_task, completed=self._wells_done
            )
            self._img_progress.update(
                self._img_task, completed=n_images, description=f"{well_pos} ✓"
            )
            self._refresh()

    def image_done(self) -> None:
        """Advance the image counter by one."""
        self._img_done += 1
        self._img_progress.update(self._img_task, completed=self._img_done)
        self._refresh()

    def set_stage(self, stage: str) -> None:
        """Update the displayed pipeline stage label.

        Args:
            stage: Short description, e.g. ``"segmentation"`` or
                ``"feature extraction"``.
        """
        self._stage = stage
        self._refresh()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eta_str(self) -> str:
        remaining = self._n_wells - self._wells_done
        if remaining <= 0 or not self._well_times:
            return "—"
        avg = sum(self._well_times) / len(self._well_times)
        secs = int(avg * remaining)
        if secs < 60:
            return f"~{secs}s"
        return f"~{secs // 60}m {secs % 60:02d}s"

    def _speed_str(self) -> str:
        elapsed = time.monotonic() - self._plate_start
        if elapsed < 1 or self._wells_done == 0:
            return "—"
        rate = self._wells_done / elapsed * 60  # wells per minute
        return f"{rate:.1f} wells/min"

    def _render(self) -> Panel:
        grid = Table.grid(padding=(0, 1))
        grid.add_column()

        # Well progress bar
        grid.add_row(self._well_progress)
        # Image progress bar
        grid.add_row(self._img_progress)
        # Status line
        status = Text()
        status.append("Stage: ", style="dim")
        status.append(f"{self._stage}", style="bold yellow")
        status.append("  •  ETA: ", style="dim")
        status.append(self._eta_str(), style="bold")
        status.append(f"  •  {self._speed_str()}", style="dim")
        grid.add_row(status)

        return Panel(
            grid,
            title=f"[bold]OMERO-Screen[/]  plate [cyan]{self._plate_id}[/]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _refresh(self) -> None:
        self._live.update(self._render())


# ---------------------------------------------------------------------------
# Tqdm-compatible redirect so Cellpose's internal bars are suppressed
# ---------------------------------------------------------------------------


class _SilentTqdm:
    """Drop-in tqdm replacement that discards all output.

    Passes the iterable through unchanged so existing ``for x in tqdm(...)``
    loops continue to work without modification.
    """

    def __init__(
        self, iterable: Iterable[object] | None = None, **kwargs: object
    ) -> None:
        self._it = iterable

    def __iter__(self) -> Iterator[object]:
        return iter(self._it or [])

    def update(self, n: int = 1) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_SilentTqdm":
        return self

    def __exit__(self, *_: object) -> None:
        pass

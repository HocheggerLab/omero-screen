"""Unified session manager dialog for managing annotation sessions.

This module provides the AnnotationSessionManager dialog that combines session
browsing, loading, file validation, and class distribution statistics into a
single unified interface.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omero_screen.config import get_logger
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.session_data_loader import SessionDataLoader

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData
    from omero_screen_napari.trainingdata_db.database import TrainingDB

logger = get_logger(__name__)


class AnnotationSessionManager(QDialog):  # type: ignore[misc]
    """Unified dialog for managing annotation sessions.

    Combines session browsing, file validation, class distribution stats,
    and session loading into a single dialog. Shows:
    - Summary stats (total sessions, cells annotated, plates, last updated)
    - Table with session details, class distributions, and file status
    - Actions: Load session, Add new data from OMERO, Refresh
    """

    def __init__(
        self,
        classifier_name: str,
        db: "TrainingDB",
        omero_data: "OmeroData",
        user_data: "UserData",
        on_session_loaded_callback: "Callable[[], None] | None" = None,
        on_direct_load_callback: "Callable[[], None] | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the session manager dialog.

        Args:
            classifier_name: Name of the classifier
            db: TrainingDB instance
            omero_data: OmeroData singleton
            user_data: UserData singleton
            on_session_loaded_callback: Callback when session is loaded
            on_direct_load_callback: Callback when new data is loaded via OMERO
            parent: Parent widget
        """
        super().__init__(parent)
        self.classifier_name = classifier_name
        self.db = db
        self.omero_data = omero_data
        self.user_data = user_data
        self.on_session_loaded_callback = on_session_loaded_callback
        self.on_direct_load_callback = on_direct_load_callback

        self.setWindowTitle(f"Manage Sessions - {classifier_name}")
        self.setMinimumSize(1100, 600)

        main_layout = QVBoxLayout()

        # Summary header
        self.summary_label = QLabel()
        self._update_summary()
        main_layout.addWidget(self.summary_label)

        # Action buttons row
        action_layout = QHBoxLayout()
        self.add_data_button = QPushButton("Add New Data")
        self.add_data_button.clicked.connect(self._on_add_new_data)
        action_layout.addWidget(self.add_data_button)

        self.export_button = QPushButton("Export Dataset")
        self.export_button.clicked.connect(self._on_export_dataset)
        action_layout.addWidget(self.export_button)

        action_layout.addStretch()

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._refresh)
        action_layout.addWidget(self.refresh_button)

        main_layout.addLayout(action_layout)

        # Sessions table
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            [
                "#",
                "Plate ID",
                "Well",
                "Images",
                "Timepoint",
                "Cell Cycle",
                "Cells",
                "Class Distribution",
                "Status",
                "Actions",
            ]
        )

        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)

        self._load_sessions()

        # Resize columns
        header = self.table.horizontalHeader()
        if header:
            for i in range(7):
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
            # Class Distribution stretches
            header.setSectionResizeMode(7, QHeaderView.Stretch)
            header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(9, QHeaderView.ResizeToContents)

        main_layout.addWidget(self.table)

        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_layout.addWidget(close_button)
        main_layout.addLayout(close_layout)

        self.setLayout(main_layout)

    def _update_summary(self) -> None:
        """Update the summary header label."""
        try:
            sessions = self.db.list_sessions(self.classifier_name)
            n_sessions = len(sessions) if sessions else 0
            n_annotations = self.db.get_total_annotations(self.classifier_name)

            plates: set[int] = set()
            if sessions:
                for session in sessions:
                    plates.add(session["plate_id"])

            last_updated = self._get_last_updated(sessions)

            self.summary_label.setText(
                f"<b>Total:</b> {n_sessions} sessions | "
                f"<b>{n_annotations}</b> cells | "
                f"<b>{len(plates)}</b> plates | "
                f"<b>Updated:</b> {last_updated}"
            )
        except Exception as e:
            logger.exception(f"Failed to create summary: {e}")
            self.summary_label.setText(f"Error loading summary: {e}")

    def _get_last_updated(self, sessions: list[dict[str, Any]] | None) -> str:
        """Calculate a human-readable 'last updated' string from session files.

        Args:
            sessions: List of session dicts (may be None/empty)

        Returns:
            Human-readable time string like "2h ago" or "Never"
        """
        if not sessions:
            return "Never"
        try:
            most_recent_time: float | None = None
            for session in sessions:
                file_path = Path(session["file_path"])
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    if most_recent_time is None or mtime > most_recent_time:
                        most_recent_time = mtime

            if most_recent_time is None:
                return "Never"

            diff = datetime.now() - datetime.fromtimestamp(most_recent_time)
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60}m ago"
            else:
                return "Just now"
        except Exception as e:
            logger.warning(f"Could not determine last updated time: {e}")
            return "Unknown"

    def _load_sessions(self) -> None:
        """Load all sessions into the table, merging list_sessions and get_image_stats."""
        try:
            sessions = self.db.list_sessions(self.classifier_name)
            stats = self.db.get_image_stats(self.classifier_name)
        except Exception as e:
            logger.exception(f"Failed to query sessions: {e}")
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {e!s}"))
            return

        if not sessions:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("No sessions available"))
            return

        # Build lookup from session_id -> full stat entry
        stats_lookup: dict[int, dict[str, Any]] = {}
        for stat in stats:
            stats_lookup[stat["session_id"]] = stat

        # Populate table
        self.table.setRowCount(len(sessions))
        for row_idx, session in enumerate(sessions):
            # Row number (1-based)
            row_item = QTableWidgetItem()
            row_item.setData(Qt.ItemDataRole.DisplayRole, row_idx + 1)  # type: ignore
            row_item.setData(Qt.ItemDataRole.UserRole, session["id"])  # type: ignore
            self.table.setItem(row_idx, 0, row_item)

            # Plate ID
            plate_item = QTableWidgetItem()
            plate_item.setData(
                Qt.ItemDataRole.DisplayRole, session["plate_id"]
            )  # type: ignore
            self.table.setItem(row_idx, 1, plate_item)

            # Well
            well = session.get("well") or "N/A"
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(well)))

            # Images (prefer image_input string from metadata, fall back
            # to numeric image_id for sessions created before this change)
            session_metadata = session.get("metadata") or {}
            image_display = session_metadata.get("image_input")
            if not image_display:
                image_id = session.get("image_id")
                image_display = (
                    str(image_id) if image_id is not None else "N/A"
                )
            image_item = QTableWidgetItem(str(image_display))
            self.table.setItem(row_idx, 3, image_item)

            # Timepoint
            timepoint = session.get("timepoint")
            tp_item = QTableWidgetItem()
            if timepoint is not None:
                tp_item.setData(Qt.ItemDataRole.DisplayRole, timepoint)  # type: ignore
            else:
                tp_item.setText("N/A")
            self.table.setItem(row_idx, 4, tp_item)

            # Cell Cycle (from session metadata user_data)
            cellcycle = "All"
            ud = session_metadata.get("user_data")
            if isinstance(ud, dict):
                cellcycle = ud.get("cellcycle", "All") or "All"
            self.table.setItem(row_idx, 5, QTableWidgetItem(cellcycle))

            # Cells + Class Distribution (from get_image_stats)
            stat_entry = stats_lookup.get(session["id"], {})
            total_cells = stat_entry.get("total_cells", 0)
            cells_item = QTableWidgetItem()
            cells_item.setData(Qt.ItemDataRole.DisplayRole, total_cells)  # type: ignore
            self.table.setItem(row_idx, 6, cells_item)

            dist = stat_entry.get("class_distribution", {})
            if dist:
                dist_str = ", ".join(
                    f"{label}: {count}" for label, count in dist.items()
                )
            else:
                dist_str = "-"
            self.table.setItem(row_idx, 7, QTableWidgetItem(dist_str))

            # Status (file validation)
            valid, msg = SessionDataLoader.validate_session_file(session)
            status_item = QTableWidgetItem(
                "\u2713 Valid" if valid else f"\u2717 {msg}"
            )
            if valid:
                status_item.setForeground(Qt.darkGreen)  # type: ignore
            else:
                status_item.setForeground(Qt.red)  # type: ignore
            self.table.setItem(row_idx, 8, status_item)

            # Actions (Load + Delete buttons)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)

            load_button = QPushButton("Load")
            load_button.clicked.connect(
                lambda checked, sid=session["id"]: self._on_load_session(sid)
            )
            actions_layout.addWidget(load_button)

            delete_button = QPushButton("Delete")
            delete_button.setStyleSheet("color: red;")
            delete_button.clicked.connect(
                lambda checked, sid=session["id"]: self._on_delete_session(sid)
            )
            actions_layout.addWidget(delete_button)

            self.table.setCellWidget(row_idx, 9, actions_widget)

    def _refresh(self) -> None:
        """Refresh the summary and sessions table."""
        self._update_summary()
        self._load_sessions()

    def _on_load_session(self, session_id: int) -> None:
        """Load a session into the viewer.

        Args:
            session_id: ID of the session to load
        """
        # Find the row for this session to check file status
        for row in range(self.table.rowCount()):
            row_item = self.table.item(row, 0)
            if (
                row_item
                and row_item.data(Qt.ItemDataRole.UserRole) == session_id
            ):  # type: ignore
                status_item = self.table.item(row, 8)
                if status_item and not status_item.text().startswith("\u2713"):
                    QMessageBox.warning(
                        self,
                        "Invalid File",
                        f"Cannot load session: {status_item.text().replace(chr(0x2717) + ' ', '')}",
                    )
                    return
                break

        try:
            success, message = SessionDataLoader.load_session(
                session_id=session_id,
                db=self.db,
                omero_data=self.omero_data,
                user_data=self.user_data,
            )

            if success:
                logger.info(f"Successfully loaded session {session_id}")
                QMessageBox.information(
                    self, "Success", f"Session loaded: {message}"
                )
                self.accept()
                if self.on_session_loaded_callback:
                    self.on_session_loaded_callback()
            else:
                logger.error(f"Failed to load session {session_id}: {message}")
                QMessageBox.critical(
                    self, "Load Failed", f"Could not load session: {message}"
                )

        except Exception as e:
            logger.exception(f"Unexpected error loading session: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Unexpected error loading session: {e!s}",
            )

    def _on_delete_session(self, session_id: int) -> None:
        """Delete a session after user confirmation.

        Removes the session from the database, deletes the NPY file on disk,
        and if this was the last session for the classifier, removes the
        entire classifier (DB record + folder on disk).

        Args:
            session_id: ID of the session to delete
        """
        # Fetch session details before deletion
        session = self.db.get_session_by_id(session_id)
        if not session:
            QMessageBox.warning(
                self, "Not Found", f"Session {session_id} not found."
            )
            return

        file_path = Path(session["file_path"])

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Session",
            f"Are you sure you want to delete this session?\n\n"
            f"Plate: {session['plate_id']}, Well: {session['well']}, "
            f"Image: {session['image_id']}\n"
            f"File: {file_path.name}\n\n"
            f"This will permanently delete the session and its NPY file.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            # Check if other sessions reference the same file before we
            # decide to delete it from disk.
            all_sessions = self.db.list_sessions(self.classifier_name)
            other_refs = [
                s
                for s in all_sessions
                if s["id"] != session_id and Path(s["file_path"]) == file_path
            ]

            # Delete session from DB (cascades to annotations)
            self.db.delete_session(session_id)
            logger.info(f"Deleted session {session_id} from database")

            # Only delete the NPY file if no other session references it
            if file_path.exists() and not other_refs:
                file_path.unlink()
                logger.info(f"Deleted NPY file: {file_path}")
            elif other_refs:
                logger.info(
                    f"Kept NPY file {file_path} — still referenced by "
                    f"{len(other_refs)} other session(s)"
                )

            # Check if this was the last session for the classifier
            remaining = self.db.get_session_count(self.classifier_name)
            if remaining == 0:
                self._delete_entire_classifier()

            self._refresh()

        except Exception as e:
            logger.exception(f"Failed to delete session {session_id}: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to delete session: {e!s}"
            )

    def _delete_entire_classifier(self) -> None:
        """Delete the entire classifier when no sessions remain.

        Removes the DB record and the classifier folder on disk
        (which contains metadata.json and any remaining files).
        """
        import shutil

        classifier_dir = (
            Path.home() / "omeroscreen_trainingdata" / self.classifier_name
        )

        reply = QMessageBox.question(
            self,
            "Delete Classifier",
            f"No sessions remain for '{self.classifier_name}'.\n\n"
            f"Delete the entire classifier and its folder on disk?\n"
            f"{classifier_dir}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Delete from DB
        self.db.delete_classifier(self.classifier_name)
        logger.info(
            f"Deleted classifier '{self.classifier_name}' from database"
        )

        # Delete folder on disk
        if classifier_dir.exists():
            shutil.rmtree(classifier_dir)
            logger.info(f"Deleted classifier folder: {classifier_dir}")

        QMessageBox.information(
            self,
            "Classifier Deleted",
            f"Classifier '{self.classifier_name}' and all its data have been removed.",
        )
        self.accept()

    def _on_export_dataset(self) -> None:
        """Build rois.npz and train.txt in the classifier directory."""
        import argparse
        import json

        import numpy as np
        from cellclass.bin.create_dataset import run as create_dataset_run

        classifier_dir = (
            Path.home() / "omeroscreen_trainingdata" / self.classifier_name
        )

        # Read expected channel count from metadata.json
        meta_path = classifier_dir / "metadata.json"
        expected_channels: int | None = None
        if meta_path.exists():
            try:
                with meta_path.open() as f:
                    meta = json.load(f)
                channels = meta.get("user_data", {}).get("channels", [])
                if channels:
                    expected_channels = len(channels)
            except Exception:
                pass

        # Pre-validate all session .npy files before running create_dataset
        if expected_channels is not None:
            sessions = self.db.list_sessions(self.classifier_name)
            bad_files: list[str] = []
            for session in sessions or []:
                fp = Path(session["file_path"])
                if not fp.exists():
                    continue
                try:
                    d = np.load(fp, allow_pickle=True).item()
                    imgs = d["data"][0]
                    if imgs and imgs[0].ndim >= 3:
                        n_ch = imgs[0].shape[-1]
                        if n_ch != expected_channels:
                            bad_files.append(
                                f"  {fp.name}  ({n_ch} channels, expected {expected_channels})"
                            )
                except Exception:
                    pass

            if bad_files:
                file_list = "\n".join(bad_files)
                QMessageBox.critical(
                    self,
                    "Channel Mismatch",
                    f"The following session files have a different number of channels "
                    f"than the classifier metadata ({expected_channels} channels):\n\n"
                    f"{file_list}\n\n"
                    "These sessions were likely saved by an older version of the widget "
                    "before channel filtering was applied. Please delete them from the "
                    "Session Manager and re-annotate if needed.",
                )
                return

        progress = QProgressDialog("Building dataset…", None, 0, 0, self)
        progress.setWindowTitle("Export Dataset")
        progress.setWindowModality(Qt.WindowModal)  # type: ignore[attr-defined]
        progress.setMinimumDuration(0)
        progress.show()

        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        try:
            args = argparse.Namespace(
                dir=str(classifier_dir),
                name="rois",
                out=None,
                ignore=["unassigned"],
                duplicates=False,
                channels=None,
                single_label=True,
            )
            create_dataset_run(args)

            npz_path = classifier_dir / "rois.npz"
            train_txt_path = classifier_dir / "train.txt"
            self._write_train_txt(train_txt_path, npz_path)

        except SystemExit as e:
            progress.close()
            logger.error(f"Dataset export failed: {e}")
            QMessageBox.critical(
                self,
                "Export Failed",
                "Dataset creation failed — check the terminal output for details.",
            )
            return
        except Exception as e:
            progress.close()
            logger.exception(f"Dataset export failed: {e}")
            QMessageBox.critical(self, "Export Failed", str(e))
            return

        progress.close()
        QMessageBox.information(
            self,
            "Export Complete",
            f"Dataset saved to:\n  {classifier_dir / 'rois.npz'}\n\n"
            f"Training template saved to:\n  {classifier_dir / 'train.txt'}\n\n"
            "Copy both files to your GPU server and run:\n"
            "  cellclass-batch train.txt --script batch.sh\n"
            "  bash batch.sh",
        )

    def _write_train_txt(self, train_txt_path: Path, npz_path: Path) -> None:
        """Write a default training sweep template.

        Uses the npz filename as a relative path so the file works from the
        classifier directory on the GPU server.

        Args:
            train_txt_path: Destination path for train.txt
            npz_path: Path to the generated rois.npz
        """
        npz_rel = npz_path.name
        flags = "--lr-scheduler plateau --flip 3 -d cuda --cudnn-benchmark --pin-memory --num-workers 4"
        lines = [
            f"{npz_rel} --model efficientnetb3s --freeze-weights --lr 1e-3 {flags} --batch-size 64",
            f"{npz_rel} --model efficientnetb3s --freeze-weights --lr 1e-4 {flags} --batch-size 64",
            f"{npz_rel} --model densenet121 --lr 1e-3 {flags} --batch-size 64",
            f"{npz_rel} --model densenet121 --lr 1e-4 {flags} --batch-size 64",
            f"{npz_rel} --model shufflenet2x1_0 --lr 1e-3 {flags} --batch-size 128",
            f"{npz_rel} --model squeezenet1_0 --lr 1e-3 {flags} --batch-size 128",
        ]
        train_txt_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote training template: {train_txt_path}")

    def _on_add_new_data(self) -> None:
        """Open dialog to add new data from OMERO."""
        try:
            from omero_screen_napari._direct_load_dialog import (
                DirectLoadDialog,
            )

            dialog = DirectLoadDialog(
                classifier_name=self.classifier_name,
                db=self.db,
                omero_data=self.omero_data,
                user_data=self.user_data,
                on_load_callback=self.on_direct_load_callback,
                parent=self,
            )
            dialog.exec_()

            if dialog.result() == QDialog.Accepted:
                self._refresh()

        except Exception as e:
            logger.exception(f"Failed to open direct load dialog: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Add New Data dialog: {e!s}",
            )

"""Session data loader for loading annotation sessions from NPY files.

This module provides the SessionDataLoader class that handles loading
previously saved annotation sessions from NPY files and populating the
OmeroData and UserData singletons.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from omero_screen.config import get_logger

from omero_screen_napari.session_utils import parse_npy_file

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData
    from omero_screen_napari.trainingdata_db.database import TrainingDB

logger = get_logger(__name__)


class SessionDataLoader:
    """Handles loading annotation sessions from NPY files into OmeroData.

    This class provides methods to load previously saved training data
    sessions, validate the NPY files, restore metadata, and populate
    the OmeroData and UserData singletons for use in the napari viewer.
    """

    @staticmethod
    def load_session(
        session_id: int,
        db: "TrainingDB",
        omero_data: "OmeroData",
        user_data: "UserData",
    ) -> tuple[bool, str]:
        """Load session data from NPY file.

        This method:
        1. Retrieves session metadata from database
        2. Validates NPY file exists
        3. Loads NPY data and populates OmeroData
        4. Restores UserData from session metadata
        5. Returns success status and message

        Args:
            session_id: ID of the session to load
            db: TrainingDB instance
            omero_data: OmeroData singleton to populate
            user_data: UserData singleton to populate

        Returns:
            Tuple of (success: bool, message: str)
            - success: True if session loaded successfully
            - message: Success message or error description

        Examples:
            >>> success, msg = SessionDataLoader.load_session(
            ...     session_id=123,
            ...     db=training_db,
            ...     omero_data=omero_data,
            ...     user_data=user_data
            ... )
            >>> if success:
            ...     print(f"Loaded: {msg}")
            ... else:
            ...     print(f"Error: {msg}")
        """
        # 1. Get session from database
        session = db.get_session_by_id(session_id)
        if not session:
            error_msg = f"Session {session_id} not found in database"
            logger.error(error_msg)
            return False, error_msg

        # 2. Validate file exists
        file_path = Path(session["file_path"])
        if not file_path.exists():
            error_msg = f"NPY file not found: {file_path}"
            logger.error(error_msg)
            return False, error_msg

        # 3. Restore UserData from session metadata BEFORE loading NPY,
        # so parse_npy_file uses the correct no_background/channels settings.
        if session.get("metadata"):
            metadata = session["metadata"]
            if isinstance(metadata, dict) and "user_data" in metadata:
                try:
                    user_data.populate_from_dict(metadata["user_data"])
                    logger.info("Restored UserData from session metadata")
                except Exception as e:
                    logger.warning(f"Could not restore UserData: {e}")
                    # Non-fatal - continue with default user_data
            # Restore channel_data so channel names can be resolved
            if "channel_data" in metadata and not omero_data.channel_data:
                omero_data.channel_data = metadata["channel_data"]
                logger.info("Restored channel_data from session metadata")

        # 4. Load and validate NPY data
        try:
            parse_npy_file(file_path, omero_data, user_data)
        except FileNotFoundError as e:
            error_msg = f"NPY file not found: {e}"
            logger.error(error_msg)
            return False, error_msg
        except ValueError as e:
            error_msg = f"Invalid NPY structure: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Cannot read NPY file: {e}"
            logger.error(error_msg)
            return False, error_msg

        # 5. Store session context in OmeroData
        omero_data.plate_id = session["plate_id"]

        # Populate well_pos_list for TrainingWidget compatibility
        if session.get("well"):
            omero_data.well_pos_list = [session["well"]]
        else:
            omero_data.well_pos_list = [""]

        # Derive image_input from the NPY filename so _set_paths() in
        # TrainingWidget produces the same filename as the saved file.
        # Format: {plate}_{well}_{image_input}_{timepoint}.npy
        stem = file_path.stem  # e.g. "1550_A2_1, 2_0"
        parts = stem.split("_", 2)  # ["1550", "A2", "1, 2_0"]
        if len(parts) >= 3:
            # Remove trailing timepoint: "1, 2_0" → "1, 2"
            image_and_tp = parts[2]
            last_underscore = image_and_tp.rfind("_")
            if last_underscore >= 0:
                omero_data.image_input = image_and_tp[:last_underscore]
            else:
                omero_data.image_input = image_and_tp
        elif session.get("image_id"):
            omero_data.image_input = str(session["image_id"])
        else:
            omero_data.image_input = ""

        logger.info(
            f"Loaded session {session_id}: plate {session['plate_id']}, "
            f"well {session.get('well', 'N/A')}, "
            f"{len(omero_data.selected_images or [])} cells"
        )

        success_msg = f"Loaded {len(omero_data.selected_images or [])} cells from session"
        return True, success_msg

    @staticmethod
    def validate_session_file(session: dict[str, Any]) -> tuple[bool, str]:
        """Validate that a session's NPY file exists and is readable.

        This is a lightweight check that doesn't load the full NPY data.
        Useful for validating sessions before attempting to load them.

        Args:
            session: Session dictionary from database (must have 'file_path' key)

        Returns:
            Tuple of (valid: bool, message: str)

        Examples:
            >>> session = db.get_session_by_id(123)
            >>> valid, msg = SessionDataLoader.validate_session_file(session)
            >>> if not valid:
            ...     print(f"Cannot load: {msg}")
        """
        if "file_path" not in session:
            return False, "Session missing file_path"

        file_path = Path(session["file_path"])
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        if not file_path.is_file():
            return False, f"Not a file: {file_path}"

        if file_path.suffix != ".npy":
            return False, f"Not a .npy file: {file_path}"

        # Check if file is readable
        try:
            with file_path.open("rb"):
                pass
        except PermissionError:
            return False, f"Permission denied: {file_path}"
        except Exception as e:
            return False, f"Cannot read file: {e}"

        return True, "File valid"

"""Parse and manage metadata for OMERO plates.

This module provides functionality to extract and validate metadata for OMERO plates.
It first checks for an attached Excel file on the plate. If found, metadata is parsed
from the Excel file; otherwise, it is extracted from existing plate annotations.

- Channel data is added to the plate as annotations.
- Well data is added as key-value annotations to each well.
- All parsed metadata is stored in a dataclass.

If no valid metadata is found, the program exits with an error.
"""

from collections import Counter
from typing import Any

from omero.gateway import BlitzGateway, FileAnnotationWrapper, PlateWrapper
from omero_utils.attachments import (
    delete_excel_attachment,
    get_file_attachments,
    parse_excel_data,
)
from omero_utils.map_anns import (
    add_map_annotations,
    delete_map_annotations,
    parse_annotations,
)
from omero_utils.message import (
    ChannelAnnotationError,
    ExcelParsingError,
    MetadataValidationError,
    PlateNotFoundError,
    WellAnnotationError,
    log_success,
)
from rich.panel import Panel
from rich.table import Table

from omero_screen.config import get_console, get_logger
from omero_screen.constants import OmeroScreenNS

logger = get_logger(__name__)

SUCCESS_STYLE = "bold cyan"

NUCLEUS_ALIASES: frozenset[str] = frozenset(
    {"dapi", "hoechst", "dna", "h2b_rfp"}
)
CELL_LEGACY_SUBSTRING: str = "Tub"
ROLE_SUFFIXES: dict[str, str] = {"_nucleus": "nucleus", "_cell": "cell"}


def strip_role_suffix(channel_name: str) -> str:
    """Return the channel name with any trailing `_nucleus` / `_cell` role suffix removed.

    Matching is case-insensitive; the rest of the name is preserved verbatim.
    Used to build feature column names without introducing duplicated suffix tokens
    (e.g. avoids ``intensity_mean_CellMask_cell_cell``).
    """
    lowered = channel_name.lower()
    for suffix in ROLE_SUFFIXES:
        if lowered.endswith(suffix):
            return channel_name[: -len(suffix)]
    return channel_name


def resolve_channel_roles(channel_data: dict[str, int]) -> dict[str, str]:
    """Resolve channel role assignments (``nucleus`` / ``cell``) from channel names.

    Resolution rules, in priority order:

    1. Suffix ``_nucleus`` (case-insensitive) → role ``nucleus``.
    2. Suffix ``_cell`` (case-insensitive) → role ``cell``.
    3. Name (lower-cased) in :data:`NUCLEUS_ALIASES` → role ``nucleus``.
    4. Name contains ``"Tub"`` (legacy) → role ``cell``.
    5. Otherwise → no role (channel is a feature channel only).

    Args:
        channel_data: Mapping of channel name → channel index. Names are preserved
            verbatim; this function does not mutate the input.

    Returns:
        Mapping ``{role: channel_name}`` containing at most one entry per role.
        Keys are a subset of ``{"nucleus", "cell"}``.

    Raises:
        ChannelAnnotationError: If no nucleus role can be resolved, or if two
            channels resolve to the same role.
    """
    roles: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {"nucleus": [], "cell": []}

    for name in channel_data:
        role = _classify_channel(name)
        if role is None:
            continue
        conflicts[role].append(name)

    for role, names in conflicts.items():
        if len(names) > 1:
            raise ChannelAnnotationError(
                f"Multiple channels resolve to role '{role}': {names}. "
                f"Only one channel per role is allowed. "
                f"Rename channels or remove the conflicting suffix.",
                logger,
            )
        if names:
            roles[role] = names[0]

    if "nucleus" not in roles:
        raise ChannelAnnotationError(
            f"No nucleus channel found in {list(channel_data)}. "
            f"Expected a channel named {sorted(NUCLEUS_ALIASES)} (case-insensitive), "
            f"or any name with the suffix '_nucleus'.",
            logger,
        )

    return roles


def _classify_channel(name: str) -> str | None:
    """Return the role of a single channel name, or None if it has no role."""
    lowered = name.lower()
    for suffix, role in ROLE_SUFFIXES.items():
        if lowered.endswith(suffix):
            return role
    if lowered in NUCLEUS_ALIASES:
        return "nucleus"
    if CELL_LEGACY_SUBSTRING in name:
        return "cell"
    return None


def _normalize_cell_line_column(df: Any) -> Any:
    """Rename any case variant of 'cell_line' to the canonical 'cell_line'.

    Args:
        df: A pandas DataFrame from the Excel metadata sheet.

    Returns:
        The DataFrame with the column renamed if a match was found.

    Raises:
        ExcelParsingError: If no cell_line column is found (any case).
    """
    col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    for original, normalized in col_map.items():
        if normalized == "cell_line" and original != "cell_line":
            logger.debug(
                "Normalizing Excel column '%s' → 'cell_line'", original
            )
            return df.rename(columns={original: "cell_line"})
    if "cell_line" not in df.columns:
        raise ExcelParsingError(
            "No 'cell_line' column found in Excel Sheet2 (checked all case variants)",
            logger,
        )
    return df


def _normalize_cell_line_key(annotation: dict[str, Any]) -> dict[str, Any]:
    """Rename any case variant of 'cell_line' key to the canonical 'cell_line'.

    Args:
        annotation: A dictionary of well annotations.

    Returns:
        The dictionary with the key renamed if a match was found.
    """
    if "cell_line" in annotation:
        return annotation
    for key in list(annotation):
        if key.lower().replace(" ", "_") == "cell_line":
            logger.debug("Normalizing annotation key '%s' → 'cell_line'", key)
            annotation["cell_line"] = annotation.pop(key)
            return annotation
    return annotation


# --------------------Metadata Parser--------------------
class MetadataParser:
    """Parses and manages channel and well metadata for an OMERO plate.

    This class extracts metadata from an OMERO plate, either from an attached Excel file or from existing plate annotations. It validates, normalizes, and stores channel and well metadata, and provides methods to annotate the plate and its wells accordingly.

    Attributes:
        conn (BlitzGateway): The OMERO connection object.
        plate_id (int): The ID of the plate to parse.
        excel_file (bool): Whether an Excel file was found and used for metadata.
        channel_data (dict[str, str]): Channel metadata, mapping channel names to indices.
            Note: during the channel-role transition, the nucleus channel is still renamed
            to ``"DAPI"`` regardless of the user-provided name. New code should look up
            the nucleus channel via ``channel_data[channel_roles["nucleus"]]`` instead.
        channel_roles (dict[str, str]): Mapping of role (``"nucleus"``, ``"cell"``) to the
            channel name in :attr:`channel_data`. Populated by :func:`resolve_channel_roles`.
        well_data (dict[str, Any]): Well metadata, mapping annotation keys to lists of values.
        pixel_size (float): Pixel size in micrometers, determined from the first image.
    """

    def __init__(self, conn: BlitzGateway, plate_id: int):
        """Initializes the MetadataParser with an OMERO connection and plate ID.

        Args:
            conn (BlitzGateway): The OMERO connection object.
            plate_id (int): The ID of the plate to parse metadata from.
        """
        self.conn: BlitzGateway = conn
        self.plate_id: int = plate_id
        self._check_plate()
        self.excel_file: bool = False
        self.channel_data: dict[str, str] = {}
        self.channel_roles: dict[str, str] = {}
        self.well_data: dict[str, Any] = {}
        self.pixel_size: float = 0
        self._empty_well_positions: list[str] = []

    def _check_plate(self) -> None:
        """Validate the OMERO plate object for the given plate ID.

        Raises:
            PlateNotFoundError: If no plate with the specified ID exists in OMERO.
        """
        plate = self._get_plate()
        if plate is None:
            raise PlateNotFoundError(
                f"A plate with id {self.plate_id} was not found!", logger
            )
        assert isinstance(plate, PlateWrapper)
        log_success(
            SUCCESS_STYLE, f"Found plate with id {self.plate_id}", logger
        )

    def manage_metadata(self) -> None:
        """Parse, validate, and apply metadata for the OMERO plate.

        This method orchestrates the metadata management workflow:
        - Parses metadata from an attached Excel file or from plate annotations.
        - Validates the extracted metadata.
        - Retrieves pixel size information from the first image.
        - If an Excel file was used, applies channel and well annotations to the plate and deletes the Excel file.
        - Logs the outcome of the metadata management process.
        """
        self._parse_metadata()  # checks for excel file or well data and pulls channel and well data into self.channel_data and self.well_data dictionaries
        self._validate_metadata()
        self._get_pixel_size()
        if self.excel_file:  # if excel file is found, add channel and well annotations to plate and delete excel file
            self._add_channel_annotations(self.channel_data)
            self._add_well_annotations(self.well_data)
            delete_excel_attachment(self.conn, self._get_plate())
            log_success(
                SUCCESS_STYLE,
                f"Metadata parsed from Excel file and transferred to plate {self.plate_id}",
                logger,
            )
        else:
            log_success(
                SUCCESS_STYLE,
                f"Metadata parsed from plate {self.plate_id}",
                logger,
            )

    def _parse_metadata(self) -> None:
        """Extract channel and well metadata from the OMERO plate.

        Attempts to parse metadata from an attached Excel file. If no Excel file is found,
        falls back to extracting metadata from existing plate and well annotations.
        Populates self.channel_data and self.well_data with the parsed results.

        Raises:
            ExcelParsingError: If there are issues parsing the Excel file.
            ChannelAnnotationError: If there are issues with channel annotations.
            WellAnnotationError: If there are issues with well annotations.
        """
        if file_annotations := self._check_excel_file():
            log_success(
                SUCCESS_STYLE,
                f"Found Excel file attachment on plate {self.plate_id}",
                logger,
            )
            try:
                self.channel_data, self.well_data = self._load_data_from_excel(
                    file_annotations
                )
            except Exception as e:
                raise ExcelParsingError(
                    f"Failed to parse Excel file: {str(e)}", logger
                ) from e
        else:
            try:
                self.channel_data = self._parse_channel_annotations()
                self.well_data = self._parse_well_annotations()
            except (ChannelAnnotationError, WellAnnotationError) as e:
                if isinstance(e, ChannelAnnotationError):
                    raise ChannelAnnotationError(
                        f"Failed to parse channel annotations: {str(e)}",
                        logger,
                    ) from e
                else:
                    raise WellAnnotationError(
                        f"Failed to parse well annotations: {str(e)}", logger
                    ) from e

    def _check_excel_file(self) -> FileAnnotationWrapper | None:
        """Check for an Excel file attachment on the OMERO plate.

        Returns:
            FileAnnotationWrapper | None: The Excel file annotation if exactly one is found; otherwise, None.

        Raises:
            ExcelParsingError: If multiple Excel file attachments are found on the plate.
        """
        # Plate is already validated in __init__
        file_annotations = get_file_attachments(self._get_plate(), ".xlsx")
        if file_annotations and len(file_annotations) == 1:
            self.excel_file = True
            return file_annotations[0]
        elif file_annotations and len(file_annotations) > 1:
            raise ExcelParsingError(
                "Multiple Excel files found on plate", logger
            )
        else:
            return None

    def _load_data_from_excel(
        self, file_annotations: FileAnnotationWrapper
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Parse channel and well metadata from an attached Excel file.

        Reads the provided Excel file annotation and extracts channel and well metadata
        from the expected sheets ('Sheet1' for channels, 'Sheet2' for wells).

        Args:
            file_annotations (FileAnnotationWrapper): The Excel file annotation to parse.

        Returns:
            tuple[dict[str, str], dict[str, Any]]: A tuple containing:
                - channel_data: Dictionary mapping channel names to indices.
                - well_data: Dictionary mapping annotation keys to lists of values.

        Raises:
            ExcelParsingError: If the Excel file format is invalid or missing required sheets.
        """
        meta_data = parse_excel_data(file_annotations)
        if not meta_data or list(meta_data.keys()) != ["Sheet1", "Sheet2"]:
            raise ExcelParsingError(
                "Invalid excel file format - expected Sheet1 and Sheet2",
                logger,
            )

        channel_data = {
            meta_data["Sheet1"]["Channels"][i]: str(
                meta_data["Sheet1"]["Index"][i]
            )
            for i in range(len(meta_data["Sheet1"]["Channels"]))
        }

        df = meta_data["Sheet2"]

        # Normalize cell_line column name (case-insensitive)
        df = _normalize_cell_line_column(df)

        # Separate Empty wells so they are excluded from validation and processing
        empty_mask = df["cell_line"].astype(str).str.strip() == "Empty"
        self._empty_well_positions = df.loc[empty_mask, "Well"].tolist()
        df = df[~empty_mask]

        well_data = {str(k): v for k, v in df.to_dict(orient="list").items()}

        return channel_data, well_data

    def _add_channel_annotations(self, channel_data: dict[str, str]) -> None:
        """Replace existing channel annotations on the plate with new channel data.

        Deletes any preexisting map annotations from the plate and adds the provided
        channel metadata as new map annotations.

        Args:
            channel_data (dict[str, str]): Dictionary mapping channel names to indices to be added as annotations.
        """
        delete_map_annotations(
            self.conn, self._get_plate(), ns=OmeroScreenNS.METADATA
        )
        add_map_annotations(
            self.conn,
            self._get_plate(),
            channel_data,
            ns=OmeroScreenNS.METADATA,
        )

    def _add_well_annotations(self, well_data: dict[str, Any]) -> None:
        """Replace existing well annotations with new metadata for each well in the plate.

        Iterates over all wells in the plate, deletes any preexisting map annotations,
        and adds new annotations based on the provided well metadata.

        Args:
            well_data (dict[str, Any]): Dictionary mapping annotation keys to lists of values for each well.
        """
        for well in self._get_plate().listChildren():
            delete_map_annotations(self.conn, well, ns=OmeroScreenNS.METADATA)
            well_name = well.getWellPos()
            try:
                well_index = well_data["Well"].index(well_name)
            except ValueError:
                logger.debug(
                    "Well %s not in metadata (may be marked as Empty), skipping annotation.",
                    well_name,
                )
                continue
            well_meta_data = {
                key: values[well_index]
                for key, values in well_data.items()
                if key
                != "Well"  # Skip the Well key since we don't need it in annotations
            }
            add_map_annotations(
                self.conn, well, well_meta_data, ns=OmeroScreenNS.METADATA
            )

    def _parse_channel_annotations(self) -> dict[str, str]:
        """Extract channel annotations from the plate and return as a dictionary.

        Parses map annotations from the plate and returns a dictionary mapping channel names
        to their indices. Raises an error if no channel annotations are found.

        Returns:
            dict[str, str]: Dictionary mapping channel names to their indices.

        Raises:
            ChannelAnnotationError: If no channel annotations are found on the plate.
        """
        annotations: dict[str, str] = parse_annotations(
            self._get_plate(), ns=OmeroScreenNS.METADATA
        )
        if len(annotations):
            return annotations
        else:
            raise ChannelAnnotationError(
                "No channel annotations found on plate", logger
            )

    def _parse_well_annotations(self) -> dict[str, Any]:
        """Extract well annotations from the plate and return as a dictionary.

        Iterates over all wells in the plate, collecting map annotations for each well.
        Returns a dictionary where each key is an annotation key and the value is a list
        of values for that key across all wells. Also includes a 'Well' key with the well positions.

        Returns:
            dict[str, Any]: Dictionary mapping annotation keys to lists of values for each well, including a 'Well' key for well positions.

        Raises:
            WellAnnotationError: If any well is missing annotations.
        """
        well_data: dict[str, list[Any]] = {"Well": []}

        for well in self._get_plate().listChildren():
            well_pos = well.getWellPos()

            well_annotation = parse_annotations(
                well, ns=OmeroScreenNS.METADATA
            )
            if not well_annotation:
                raise WellAnnotationError(
                    f"No well annotations found for well {well_pos}", logger
                )

            # Normalize cell_line key (case-insensitive)
            well_annotation = _normalize_cell_line_key(well_annotation)

            # Skip Empty wells entirely — they have no experimental data
            if well_annotation.get("cell_line") == "Empty":
                self._empty_well_positions.append(well_pos)
                continue

            well_data["Well"].append(well_pos)
            for key, value in well_annotation.items():
                if key not in well_data:
                    well_data[key] = []
                well_data[key].append(value)
        return well_data

    def _create_two_column_table(
        self, title: str, col1_name: str, col2_name: str
    ) -> Table:
        """Create and return a Rich table with two columns and a title.

        Args:
            title (str): The title of the table.
            col1_name (str): Name of the first column.
            col2_name (str): Name of the second column.

        Returns:
            Table: A Rich Table object with the specified columns and title.
        """
        table = Table(title=title)
        table.add_column(col1_name, style="cyan")
        table.add_column(col2_name, style="green")
        return table

    def _display_metadata(self) -> None:
        """Display parsed channel and well metadata using Rich tables.

        Formats and prints channel and well metadata in visually appealing tables using the Rich library.
        Channel data is shown as a two-column table, and well data is summarized with unique values and counts.
        """
        # Create and populate channel table
        channel_table = self._create_two_column_table(
            "Channel Information", "Channel Name", "Index"
        )
        for channel, index in self.channel_data.items():
            channel_table.add_row(channel, str(index))

        # Create and populate well summary table
        well_table = self._create_two_column_table(
            "Well Data Summary", "Key", "Unique Values"
        )
        well_table.add_column("Count", style="yellow")

        for key, values in self.well_data.items():
            if key != "Well":  # Skip the Well column as it's too verbose
                unique_values = Counter(values)
                well_table.add_row(
                    key,
                    ", ".join(str(k) for k in unique_values),
                    ", ".join(str(v) for v in unique_values.values()),
                )

        # Display the tables in panels
        get_console().print(
            Panel(channel_table, title="Channel Data", border_style="cyan")
        )
        get_console().print(
            Panel(well_table, title="Well Data Summary", border_style="cyan")
        )

    def _validate_metadata(self) -> None:
        """Validate the structure and content of parsed metadata.

        Runs a series of validation checks on channel and well metadata, collecting all errors.
        If any validation errors are found, raises a MetadataValidationError with details.
        Also displays the metadata if validation passes.

        Raises:
            MetadataValidationError: If any validation errors are detected in the metadata.
        """
        errors = []

        # Collect errors from all validation steps
        errors.extend(self._validate_metadata_structure())
        errors.extend(self._validate_channel_data())
        errors.extend(self._validate_well_data())

        # If any errors were found, raise them all together
        if errors:
            if len(errors) == 1:
                raise MetadataValidationError(
                    errors[0],
                    logger,
                )
            else:
                raise MetadataValidationError(
                    "Multiple validation errors found:\n"
                    + "\n".join(f"- {error}" for error in errors),
                    logger,
                )

        log_success(
            SUCCESS_STYLE,
            f"Metadata validation passed for plate {self.plate_id}",
            logger,
        )

        self._display_metadata()

    def _validate_metadata_structure(self) -> list[str]:
        """Check the basic structure and types of the parsed metadata.

        Validates that channel and well metadata are present and have the correct types.
        Returns a list of error messages for any structural issues found.

        Returns:
            list[str]: A list of error messages. The list is empty if no errors are found.
        """
        errors = []
        if not self.channel_data:
            errors.append("No channel data found")
        if not self.well_data:
            errors.append("No well data found")

        if not isinstance(self.channel_data, dict) or not all(
            isinstance(k, str) for k in self.channel_data
        ):
            errors.append("Channel data must be a dictionary with string keys")
        if not all(isinstance(v, str) for v in self.channel_data.values()):
            errors.append(
                "Channel data must be a dictionary with string values"
            )

        return errors

    def _validate_channel_data(self) -> list[str]:
        """Validate channel metadata and populate channel roles.

        Resolves channel roles (``nucleus`` / ``cell``) from the user-provided
        channel names into :attr:`channel_roles`, using :func:`resolve_channel_roles`.
        Channel names in :attr:`channel_data` are preserved verbatim — the actual
        fluorophore (e.g. ``"Hoechst"``, ``"H2B_RFP"``) flows through to feature
        column names so downstream analysis is biologically truthful.

        Validates that channel indices are valid integers forming a contiguous
        sequence starting from 0.

        Returns:
            list[str]: A list of error messages. The list is empty if no errors are found.
        """
        errors = []

        # Resolve channel roles from the user-provided names. The resolver
        # raises ``ChannelAnnotationError`` when no nucleus channel can be
        # identified, which we surface as a validation error.
        try:
            self.channel_roles = resolve_channel_roles(
                dict.fromkeys(self.channel_data, 0)
            )
        except ChannelAnnotationError as exc:
            errors.append(str(exc))
            self.channel_roles = {}

        # Validate channel indices are valid integers
        parsed_indices: list[int] = []
        for ch_name, ch_idx in self.channel_data.items():
            try:
                parsed_indices.append(int(ch_idx))
            except (ValueError, TypeError):
                errors.append(
                    f"Channel '{ch_name}' has non-integer index '{ch_idx}'. "
                    f"Channel indices must be integers (e.g. 0, 1, 2, ...)"
                )

        if parsed_indices:
            n_channels = len(parsed_indices)
            expected = set(range(n_channels))
            actual = set(parsed_indices)
            if actual != expected:
                errors.append(
                    f"Channel indices must be 0 to {n_channels - 1} for {n_channels} channels, "
                    f"but got {sorted(actual)}. "
                    f"Check your Excel metadata — indices should be consecutive starting from 0."
                )

        # Validate channel indices against actual image channels
        if not errors:
            errors.extend(self._validate_channel_indices_against_image())

        return errors

    def _validate_channel_indices_against_image(self) -> list[str]:
        """Validate that channel indices in metadata do not exceed the actual number of channels in the images.

        Returns:
            list[str]: A list of error messages. The list is empty if no errors are found.
        """
        errors = []
        try:
            first_image = self._get_first_image()
            actual_channel_count = first_image.getSizeC()
            for ch_name, ch_idx in self.channel_data.items():
                idx = int(ch_idx)
                if idx < 0 or idx >= actual_channel_count:
                    errors.append(
                        f"Channel '{ch_name}' has index {idx}, but images only have "
                        f"{actual_channel_count} channels (valid indices: 0-{actual_channel_count - 1}). "
                        f"Check your Excel metadata."
                    )
        except Exception:  # noqa: BLE001
            # Image retrieval may fail (e.g. no wells yet, or in unit tests).
            # Channel-vs-image validation is skipped; other checks still apply.
            logger.debug(
                "Skipping channel index vs image validation (no image available)"
            )
        return errors

    def _validate_well_data(self) -> list[str]:
        """Validate the well data structure and content.

        Returns:
            list[str]: List of error messages, empty if no errors
        """
        errors = []
        # Check required keys exist (cell_line is case-insensitive, already normalized)
        required_keys = {"Well", "cell_line"}
        if missing_keys := required_keys - self.well_data.keys():
            errors.append(
                f"Missing required keys in well data: {', '.join(missing_keys)}"
            )

        if non_list_keys := [
            key
            for key, value in self.well_data.items()
            if not isinstance(value, list)
        ]:
            errors.append(
                f"Values must be lists for all keys. Non-list values found for: {', '.join(non_list_keys)}"
            )

        # Check all lists have the same length
        list_lengths = {
            key: len(value) for key, value in self.well_data.items()
        }
        if len(set(list_lengths.values())) > 1:
            # Create a message showing the different lengths
            length_info = [
                f"{key}: {length}" for key, length in list_lengths.items()
            ]
            errors.append(
                f"All well data lists must have the same length. Found: {', '.join(length_info)}"
            )

        # Validate well positions match actual wells in plate
        errors.extend(self._validate_well_positions())

        return errors

    def _validate_well_positions(self) -> list[str]:
        """Check that well positions in metadata match the actual wells in the plate.

        Compares the well positions listed in the metadata with those present in the plate.
        Returns a list of error messages for any missing or extra wells.

        Returns:
            list[str]: A list of error messages. The list is empty if no errors are found.
        """
        errors = []
        # Get actual well positions from the plate, excluding Empty wells
        actual_wells = [
            well.getWellPos()
            for well in self._get_plate().listChildren()
            if well.getWellPos() not in self._empty_well_positions
        ]

        # Get well positions from metadata
        metadata_wells = self.well_data["Well"]

        # Check for missing and extra wells
        s1 = set(actual_wells)
        logger.debug(f"Actual wells: {s1}")
        s2 = set(metadata_wells)
        logger.debug(f"Metadata wells: {s2}")
        if s1 != s2:
            missing_wells = s1 - s2
            extra_wells = s2 - s1
            if len(missing_wells):
                errors.append(
                    f"Missing wells in metadata: {', '.join(sorted(missing_wells))}"
                )
            if len(extra_wells):
                errors.append(
                    f"Extra wells in metadata: {', '.join(sorted(extra_wells))}"
                )

        # Here the plate wells and the metadata have the same well position names.
        # The order does not matter as the dictionary list under 'Well' is used to index
        # into the well values for each key (see method: well_conditions)

        return errors

    def _get_first_image(self) -> Any:
        """Retrieve the first image from the first well of the plate.

        Accesses the first well and its first well sample to obtain the associated image.
        Raises an error if no wells or images are found.

        Returns:
            Any: The first image object from the first well.

        Raises:
            MetadataValidationError: If no wells or images are found in the plate.
        """
        # Get the first well
        first_well = next(self._get_plate().listChildren(), None)
        if not first_well:
            raise MetadataValidationError("No wells found in plate", logger)

        # Get the first well sample from the well
        first_well_sample = next(first_well.listChildren(), None)
        if not first_well_sample:
            raise MetadataValidationError(
                "No images found in first well", logger
            )

        if first_image := first_well_sample.getImage():
            return first_image
        else:
            raise MetadataValidationError(
                "Could not get image from well sample", logger
            )

    def _get_pixel_size(self) -> None:
        """Determine the pixel size in micrometers from the first image of the first well.

        Retrieves the pixel size (X and Y) from the primary pixels of the first image.
        Validates that both pixel sizes are present and equal. Sets self.pixel_size to the value.

        Raises:
            MetadataValidationError: If pixel size information is missing, inconsistent, or cannot be determined.
        """
        first_image = self._get_first_image()

        # Get the pixel size from the image's pixels
        pixels = first_image.getPrimaryPixels()
        if not pixels:
            raise MetadataValidationError(
                "No pixel information found in image", logger
            )

        # Get the physical size in micrometers
        pixel_size_x = round(float(pixels.getPhysicalSizeX().getValue()), 1)
        pixel_size_y = round(float(pixels.getPhysicalSizeY().getValue()), 1)
        logger.debug(
            "Pixel size x: %s, Pixel size y: %s", pixel_size_x, pixel_size_y
        )
        # Validate that we have valid pixel sizes
        if pixel_size_x is None or pixel_size_y is None:
            raise MetadataValidationError(
                "Could not determine pixel size from image", logger
            )

        if pixel_size_x != pixel_size_y:
            raise MetadataValidationError(
                f"Pixel size x ({pixel_size_x}) and y ({pixel_size_y}) are not the same",
                logger,
            )

        log_success(
            SUCCESS_STYLE,
            f"The images in plate {self.plate_id} have a pixel size of {pixel_size_x} micrometers",
            logger,
        )
        self.pixel_size = pixel_size_x

    def well_conditions(self, well_id: str) -> dict[str, Any]:
        """Get the conditions for the specified well position (e.g. A1).

        See: WellWrapper.getWellPos().

        Returns:
            dict[str, Any]: Dictionary with annotations

        Raises:
            ValueError: If the well position is not found in the metadata.
        """
        try:
            idx = self.well_data["Well"].index(well_id)
        except ValueError:
            available = sorted(self.well_data["Well"])
            raise ValueError(
                f"Well '{well_id}' not found in metadata. "
                f"Available wells: {available}. "
                f"Check that the well positions in your Excel file match the plate layout."
            ) from None
        return {k: v[idx] for k, v in self.well_data.items() if k != "Well"}

    def _get_plate(self) -> PlateWrapper:
        """Get a refreshed plate object from OMERO.

        This method is used to obtain an updated plate after all modifications to the object.

        Returns:
            The OMERO plate object
        """
        return self.conn.getObject("Plate", self.plate_id)

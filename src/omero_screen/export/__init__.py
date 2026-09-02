"""Export OMERO plates as PerkinElmer Harmony measurements.

The exported folder is a plain Operetta/Harmony measurement -- TIFFs plus an
``Index.idx.xml`` -- so re-importing it needs nothing more than the ``omero
import`` command already used by ``scripts/load_plates.sh``. That makes it a
suitable published artefact for demo/paper data: a reader can inspect the TIFFs
directly and push them through the whole pipeline from scratch.

Experimental conditions have no Harmony equivalent, so they are written
alongside as a ``metadata.xlsx`` that can be re-attached to the imported plate
(see :mod:`~.metadata_sheet`).
"""

from omero_screen.export.harmony_xml import (
    ImageSpec,
    PlateSpec,
    build_index_xml,
)
from omero_screen.export.metadata_sheet import write_metadata_excel
from omero_screen.export.plate_reader import read_plate
from omero_screen.export.writer import estimate_size_bytes, write_measurement

__all__ = [
    "ImageSpec",
    "PlateSpec",
    "build_index_xml",
    "estimate_size_bytes",
    "read_plate",
    "write_measurement",
    "write_metadata_excel",
]

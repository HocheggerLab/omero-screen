"""Scratch script for ad-hoc OMERO inspection."""

# import time
# from omero.gateway import BlitzGateway
# from omero_screen_napari.omero_image import (
#     get_omero_image_wrapper, initialise_download, get_omero_image_timepoint
# )
# conn = BlitzGateway("helfrid", "Omero_21", host="ome2.hpc.sussex.ac.uk", port=4064)
# conn.connect()
# # Server version — go through the underlying client
# try:
#     print("Server version:", conn.c.getProperty("omero.version"))
# except Exception:
#     pass
# # Config properties
# config = conn.getConfigService()
# for p in [
#     "omero.threads.max_threads",
#     "omero.threads.min_threads",
#     "omero.jvmcfg.heap.server",
#     "omero.jvmcfg.system_memory",
#     "omero.data.dir",
# ]:
#     try:
#         val = config.getConfigValue(p)
#         if val:
#             print(f"{p} = {val}")
#     except Exception:
#         pass
# # ---- Bandwidth test ----
# image_id = 13259  # put a real image id here
# wrapper = get_omero_image_wrapper(conn, image_id)
# store, shape, dt_be = initialise_download(conn, wrapper)
# # Warmup call — first RPC is always slower
# _ = get_omero_image_timepoint(store, 0, shape, dt_be)
# # 5 timed runs
# times = []
# for _ in range(5):
#     t0 = time.perf_counter()
#     arr = get_omero_image_timepoint(store, 0, shape, dt_be)
#     times.append(time.perf_counter() - t0)
# store.close()
# conn.close()
# mb = arr.nbytes / 1e6
# avg = sum(times) / len(times)
# print(f"\nImage shape: {arr.shape}")
# print(f"Uncompressed size: {mb:.1f} MB")
# print(f"Transfer times: {[f'{t:.3f}s' for t in times]}")
# print(f"Average: {avg:.3f}s  →  {mb/avg:.1f} MB/s")
import os
from typing import Any

os.environ["ENV"] = "production"

from omero.gateway import MapAnnotationWrapper
from omero_utils.omero_connect import omero_connect

from omero_screen.constants import OmeroScreenNS


@omero_connect
def go(plate_id: int, conn: Any = None) -> None:
    """Inspect map annotations on a plate and clean up duplicates."""
    plate = conn.getObject("Plate", plate_id)
    me = conn.getUser().getName()
    print(f"You: {me}")
    print(
        f"Group permissions: {plate.getDetails().getGroup().getDetails().permissions}"
    )

    ann_ids = []
    for ann in plate.listAnnotations(ns=OmeroScreenNS.DATASET):
        if isinstance(ann, MapAnnotationWrapper):
            owner = ann.getDetails().getOwner().getOmeName()
            print(
                f"  ann {ann.getId()}  owner={owner}  value={dict(ann.getValue())}"
            )
            ann_ids.append(ann.getId())

    if not ann_ids:
        print("Nothing to delete")
        return

    # Synchronous delete, raises if it actually fails
    handle = conn.deleteObjects(
        "MapAnnotation", ann_ids, deleteAnns=True, wait=True
    )
    err = handle.errors if hasattr(handle, "errors") else None
    print(f"Delete result: errors={err}")


go(1237)

"""Module to classify images using a neural network model.
The model is stored in OMERO as a TorchScript file. It is downloaded
from the CNN models project using the model name to identify the file.
The model is pretrained to classify images using regions cropped from
the image channels. The channels, crop size, network input size, and
network classification labels are stored in OMERO as a corresponding
metadata JSON file."""

import json
import logging
import pathlib
from collections.abc import Sequence
from random import randrange
from typing import Any

import numpy as np
import numpy.typing as npt
import omero
import pandas as pd
import skimage.transform
import torch
from omero.gateway import BlitzGateway
from skimage.measure import regionprops
from tqdm import tqdm

logger = logging.getLogger("omero-screen")


class ImageClassifier:
    """
    Classify images using a model.
    """

    def __init__(
        self, conn: BlitzGateway, model_name: str, class_name: str = "Class"
    ):
        """
        Initialize the classifier.
        Args:
            conn: Connection to OMERO
            model_name: Name of classification model
            class_name: Name of class (column title added to a dataframe)
        """
        self.image_data: None | dict[str, npt.NDArray[Any]] = None
        self.crop_size = 0
        self.input_shape: tuple[int, ...] = ()
        self.gallery_size = 0
        self.batch_size = 16
        self.class_name = class_name
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.selected_channels: list[npt.NDArray[Any]] = []
        # list[Any] is a pair of [list[npt.NDArray[Any]], int]: image samples, total image count.
        # We use a list and not a tuple to allow the count at index 1 to be modified
        # without having to create a new tuple and put back into the dictionary.
        self.gallery_dict: dict[str, list[Any]] = {}

        self.model, self.active_channels, self.class_options = (
            self._load_model_from_omero(conn, "CNN_Models", model_name)
        )

    def _load_model_from_omero(
        self, conn: BlitzGateway, project_name: str, model_name: str
    ) -> tuple[
        torch.jit.ScriptModule | None, list[str] | None, list[str] | None
    ]:
        """
        Laod the model from OMERO.
        Args:
            conn: Connection to OMERO
            project_name: Name of project containing the models
            model_name: Name of classification model
        Returns:
            model: classification model; active_channels: list of channels used for the model; class_options: list of the classes assigned by the model
        """
        model_filename = self._download_file(
            conn, project_name, model_name, model_name + ".pt"
        )
        if not model_filename:
            return None, None, None
        meta_filename = self._download_file(
            conn, project_name, model_name, model_name + ".json"
        )
        if not meta_filename:
            return None, None, None

        # Extract image channels
        active_channels, class_options = self._extract_channels(meta_filename)

        if active_channels:
            print(f"Active Channels: {active_channels}")
        else:
            print("No active channels found.")
            return None, None, None

        # list of random samples, total number of items
        self.gallery_dict = {
            class_name: [[], 0] for class_name in class_options
        }

        # Load the model
        model = torch.jit.load(  # type: ignore[no-untyped-call]
            model_filename, map_location=torch.device("cpu")
        )
        model = model.to(self.device)
        model.eval()
        return model, active_channels, class_options

    def _download_file(
        self,
        conn: BlitzGateway,
        project_name: str,
        dataset_name: str,
        file_name: str,
    ) -> pathlib.Path | None:
        """
        Download the file attachment.
        Args:
            conn: Connection to OMERO
            project_name: The name of the project in OMERO.
            dataset_name: The name of the dataset in OMERO.
            file_name: The name of the file attachment in OMERO.
        Returns:
            Path to local file (or None).
        """
        local_path = (
            pathlib.Path.home() / ".cache" / "omero_screen" / file_name
        )

        # If the model file does not exist locally
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Find the project in OMERO
            project = conn.getObject(
                "Project", attributes={"name": project_name}
            )
            if project is None:
                logger.warning(
                    "Project '%s' not found in OMERO.", project_name
                )
                return None

            # Find the dataset in OMERO
            dataset = next(
                (
                    ds
                    for ds in project.listChildren()
                    if ds.getName() == dataset_name
                ),
                None,
            )
            if dataset is None:
                logger.warning(
                    "Dataset '%s' not found in project '%s'.",
                    dataset_name,
                    project_name,
                )
                return None

            # Check annotations in the dataset
            for attachment in dataset.listAnnotations():
                if (
                    isinstance(attachment, omero.gateway.FileAnnotationWrapper)
                    and attachment.getFileName() == file_name
                ):
                    # Download the model file
                    with open(local_path, "wb") as f:
                        for chunk in attachment.getFileInChunks():
                            f.write(chunk)
                    logger.info("Downloaded model file to %s", local_path)
                    return local_path

            logger.warning(
                "File '%s' not found in dataset '%s' under project '%s'.",
                file_name,
                dataset_name,
                project_name,
            )
            return None
        # Already cached
        return local_path

    def _extract_channels(
        self, meta_filename: pathlib.Path
    ) -> tuple[list[str], list[str]]:
        # Read the metadata.json file and extract active channels
        try:
            with open(meta_filename) as f:
                metadata = json.load(f)
                active_channels = metadata.get("channels", [])
                class_options = metadata.get("labels", [])
                img_shape = metadata.get("img_shape", None)
                input_shape = metadata.get("input_shape", None)
                if (
                    active_channels
                    and class_options
                    and img_shape
                    and input_shape
                ):
                    # assume a square image for cropping
                    self.crop_size = img_shape[-1]
                    self.input_shape = tuple(input_shape[-2:])
                    logger.info(
                        "Active channels extracted: %s", str(active_channels)
                    )
                    logger.info("Class options: %s", class_options)
                    logger.info("Crop Size: %s", self.crop_size)
                    logger.info("Input shape: %s", self.input_shape)
                    return active_channels, class_options
                else:
                    logger.warning(
                        "Metadata file '%s' does not contain required information.",
                        meta_filename,
                    )
        except Exception as e:
            logger.exception(
                "Error reading metadata file '%s'", meta_filename, e
            )
        return [], []

    def select_channels(self, image_data: dict[str, npt.NDArray[Any]]) -> bool:
        """
        Select the channels from the image_data to be used for classification.
        If this method returns False then the classifier is not able to process the images.
        Args:
            image_data: Dictionary of images keyed by channel name. Images should be [TYX].
        Returns:
            True if the channels were selected.
        """
        if self.active_channels:
            self.image_data = image_data
            self.selected_channels = [
                image_data[channel] for channel in self.active_channels
            ]
            logger.info(
                "Selected channels for classification: %s",
                str(self.active_channels),
            )
            return True
        return False

    def process_images(
        self, original_image_df: pd.DataFrame, mask: npt.NDArray[Any]
    ) -> pd.DataFrame:
        """
        For each object in the segmentation mask, eatract the image channels, mask the pixels and run
        the model to classify the object. Classifications are appended as a new column with the heading
        specified in the object constructor.
        Args:
            original_image_df: Dataframe of OMERO screen segmented cell objects.
            mask: Segmentation mask. Should be TZYX with Z=1.
        Returns:
            Dataframe supplemented with the classification column. Returned unchanged if the classifier is
            not initialised.
        """
        if len(self.selected_channels) == 0:
            return original_image_df
        # Model should be initialised
        assert self.model is not None
        assert self.class_options is not None

        predicted_classes = []

        # Entries may share the same cytoplasm ID, e.g. cells with multiple nuclei
        image_df = original_image_df.drop_duplicates(
            subset="Cyto_ID", keep="first"
        )
        logger.info(
            "Classification of %d cyto IDs (%d items)",
            len(image_df),
            len(original_image_df),
        )

        img_size = (
            self.crop_size,
            self.crop_size,
        )  # Target size (height, width)
        half_crop = self.crop_size // 2

        # Assume YX are the last dimensions
        max_length_x = self.selected_channels[0].shape[-1]
        max_length_y = self.selected_channels[0].shape[-2]

        # Batch processing
        total = len(image_df["centroid-0"])
        step = self.batch_size
        pbar = tqdm(total=total)
        for start in range(0, total, step):
            stop = min(start + step, total)
            pbar.n = stop
            pbar.refresh()

            batch = []
            for i in range(start, stop):
                # Center the crop around the centroid coordinates
                centroid_x = image_df["centroid-1_x"].iloc[i]
                centroid_y = image_df["centroid-0_y"].iloc[i]

                x0 = int(max(0, centroid_x - half_crop))
                x1 = int(min(max_length_x, centroid_x + half_crop))
                y0 = int(max(0, centroid_y - half_crop))
                y1 = int(min(max_length_y, centroid_y + half_crop))

                # Crop mask
                cropped_mask = self._crop(mask, x0, y0, x1, y1).copy()
                # Pass in the translated centroid allowing for the crop to clip
                cx = min(half_crop, int(centroid_x))
                cy = min(half_crop, int(centroid_y))
                corrected_mask = self._erase_masks(cropped_mask, cx, cy)
                # Convert mask to binary
                binary_mask = (corrected_mask > 0).astype(np.uint8)

                # Create cropped YXC image
                img = np.dstack(
                    [
                        self._crop(i, x0, y0, x1, y1)
                        for i in self.selected_channels
                    ]
                )
                # Image normalisation copied from training repo cellclass:src/bin/create_dataset.py
                # Extract (1, 99) percentile and convert to 8-bit
                img = self._to_uint8(img)
                # Remove pixels outside the mask (transforms image to CYX)
                img = self._extract_roi(img, binary_mask)
                # Pad to required image size
                img = self._add_padding(img, img_size)
                batch.append(img)

            # Create tensor (B, C, H, W)
            batch_imgs = np.array(batch)

            # Value scaling: [0,255] -> [0,1]
            tensor = np.divide(batch_imgs, 255, dtype=np.float32)
            # Resize to model input size
            output_shape = batch_imgs.shape[0:2] + self.input_shape
            tensor = skimage.transform.resize(  # type: ignore[no-untyped-call]
                tensor, output_shape, mode="edge"
            )
            # Convert to tensor in place
            classes = self._classify(torch.from_numpy(tensor))
            predicted_classes.extend(classes)

            # Optional gallery
            if self.gallery_size:
                for idx, predicted_class in enumerate(classes):
                    a = self.gallery_dict[predicted_class]
                    # list of random samples, total number of items
                    samples, total_items = a
                    total_items = total_items + 1
                    a[1] = total_items
                    # Image is CYX
                    img = batch[idx]
                    if total_items <= self.gallery_size:
                        # Gallery size not yet reached
                        samples.append(img)
                    else:
                        # Randomly replace a gallery image
                        i = randrange(total_items)
                        if i < self.gallery_size:
                            samples[i] = img

        image_df.insert(
            len(image_df.columns), self.class_name, predicted_classes
        )

        return original_image_df.merge(
            image_df[["Cyto_ID", self.class_name]], on="Cyto_ID", how="left"
        )

    def _classify(self, image_tensor: torch.Tensor) -> list[str]:
        # Validate the set-up
        assert self.model is not None, "No model initialised"
        assert self.class_options is not None, "No classes"

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            # Model takes a tensor of (B, C, H, W)
            outputs = self.model(image_tensor)
            # Find maximum of all elements along dim=1 (i.e. classification): (values, indices)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()

        class_names = self.class_options
        predicted_class = [class_names[x] for x in predicted]

        return predicted_class

    def _crop(
        self, image: npt.NDArray[Any], x0: int, y0: int, x1: int, y1: int
    ) -> npt.NDArray[Any]:
        """
        Crops the input image using the provided coordinates.
        Args:
            image: The image to be cropped.
            x0: Top-left X coordinate for cropping.
            y0: Top-left Y coordinate for cropping.
            x1: Bottom-right X coordinate for cropping.
            y1: Bottom-right Y coordinate for cropping.
        Returns:
            Cropped image as a numpy array. Note: This uses the same underlying data.
        """
        # Crop with numpy to return a 2D image with YX as last dimensions
        i = image.squeeze()
        if i.ndim != 2:
            raise Exception(
                "Image classifier only supports 2D images: " + str(image.shape)
            )
        return i[y0:y1, x0:x1]

    def _to_uint8(self, image: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
        """
        Convert image to uint8 using the (1, 99) percentiles.
        Args:
            image: The image
        Returns:
            Scaled image
        """
        out = np.zeros(image.shape, dtype=np.uint8)
        for c in range(image.shape[-1]):
            data = image[..., c]
            pmin, pmax = np.percentile(data, (1, 99))
            # Rescale 0-1
            data = (data - pmin) / (pmax - pmin)
            # Clip to range
            data[data < 0] = 0
            data[data > 1] = 1
            # Convert to [1, 255] to allow masked regions to be distinct at zero
            out[..., c] = (1 + data * 254).astype(np.uint8)
        return out

    def _extract_roi(
        self, image: npt.NDArray[Any], binary_mask: npt.NDArray[np.uint8]
    ) -> npt.NDArray[Any]:
        """
        Extracts the ROI (Region of Interest) from a multi-channel image using the mask.
        Args:
            image: Multi-channel input image (YXC).
            binary_mask: Mask image (YX).
        Returns:
            ROI (CYX)
        """
        # Apply the mask to all channels
        roi = np.zeros_like(image)
        for channel in range(image.shape[-1]):
            roi[..., channel] = image[..., channel] * binary_mask
        return roi.transpose((2, 0, 1))

    def _add_padding(
        self,
        image: npt.NDArray[Any],
        target_size: tuple[int, int],
        padding_value: int = 0,
    ) -> npt.NDArray[Any]:
        """
        Adds padding to a NumPy array image to reach the target size.
        Args:
            image: Input image as a NumPy array (H, W) or (C, H, W).
            target_size: Target size as (height, width).
            padding_value: Value to use for padding (default: 0).
        Returns:
            Padded image with the desired target size.
        """
        current_height, current_width = image.shape[-2:]
        target_height, target_width = target_size

        # Calculate the padding needed
        pad_height = max(0, target_height - current_height)
        pad_width = max(0, target_width - current_width)

        # Divide padding equally on all sides
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Determine padding configuration based on the number of dimensions
        padding_config: Sequence[tuple[int, int]]
        if image.ndim == 2:  # Grayscale image
            padding_config = ((pad_top, pad_bottom), (pad_left, pad_right))
        elif image.ndim == 3:  # RGB or multi-channel image
            padding_config = (
                (0, 0),
                (pad_top, pad_bottom),
                (pad_left, pad_right),
            )
        else:
            raise ValueError(
                "Unsupported image dimensions. Image must be 2D or 3D."
            )

        # Apply padding
        padded_image = np.pad(
            image,
            padding_config,
            mode="constant",
            constant_values=padding_value,
        )
        return padded_image

    def _erase_masks(
        self, cropped_label: npt.NDArray[Any], cx: int, cy: int
    ) -> npt.NDArray[Any]:
        """
        Erases all masks in the cropped label (yx format) that do not overlap with the centroid.
        If no label overlaps with the centroid then the closest label is used based on their centroids.
        An exception is raised if there are no labels.
        Data is modified in-place.
        Args:
            cropped_label: Input image as a NumPy array (H, W) or (C, H, W).
            cx: Centroid X
            cy: Centroid Y
        Returns:
            The input label
        Raises:
            Exception: If there are no labels (non-zero values)
        """
        # Fast option assumes overlap of centroid with a label
        label_id = cropped_label[cy, cx]
        if label_id == 0:
            # This should not happen, log it so the user can investigate
            logger.warning("No label at %d,%d", cx, cy)
            # Find closest label
            dmin = np.prod(np.array(cropped_label.shape))
            dmin = dmin**2
            for p in regionprops(cropped_label.astype(int)):  # type: ignore[no-untyped-call]
                y, x = p.centroid
                d = (cx - x) ** 2 + (cy - y) ** 2
                if d < dmin:
                    dmin = d
                    label_id = p.label
            if label_id == 0:
                raise Exception(f"No label at {cx},{cy}")

        cropped_label[cropped_label != label_id] = 0
        return cropped_label

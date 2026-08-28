import os
from typing import Tuple

import numpy as np
import pandas as pd
import pooch
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table
from skimage.transform import resize
from skimage.util import img_as_ubyte
from tqdm import tqdm
from ultralytics import YOLO

from lutuflow.core.configuration import YOLO_MODELS


def _extract_3d_roi(
    image: np.ndarray, lungs_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a 3D region of interest (ROI) from an image using a lungs mask.

    Parameters
    ----------
    image : np.ndarray
        3D image array.
    lungs_mask : np.ndarray
        3D binary mask of the lungs region.

    Returns
    -------
    roi : np.ndarray
        Cropped image array corresponding to the bounding box of the lungs mask.
    roi_mask : np.ndarray
        Binary mask of the lungs region within the cropped image.
    """
    df = pd.DataFrame(
        regionprops_table(
            lungs_mask,
            intensity_image=image,
            properties=["bbox", "image"],
        )
    )

    # We assume a single object is present int he lungs mask
    x0 = int(df["bbox-0"].values[0])
    x1 = int(df["bbox-3"].values[0])
    y0 = int(df["bbox-1"].values[0])
    y1 = int(df["bbox-4"].values[0])
    z0 = int(df["bbox-2"].values[0])
    z1 = int(df["bbox-5"].values[0])

    roi = image[x0:x1, y0:y1, z0:z1]
    roi_mask = df["image"][0]
    roi_mask = img_as_ubyte(roi_mask)  # Convert bool to uint8

    return roi, roi_mask


def _keep_biggest_object(lab_int: np.ndarray) -> np.ndarray:
    """Selects only the biggest object of a labels image."""
    if lab_int.sum() == 0:
        return lab_int
    
    labels = ndi.label(lab_int)[0]
    counts = np.unique(labels, return_counts=1)
    biggestLabel = np.argmax(counts[1][1:]) + 1

    return (labels == biggestLabel).astype(np.uint8)


### For V1 ###

def _handle_2d_predict(image, model):
    image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    image = gray2rgb(image)

    result = model.predict(
        source=image,
        conf=0.1,  # Confidence threshold for detections.
        iou=0.5,  # Intersection over union threshold.
        imgsz=640,  # Square resizing
        max_det=2,  # Two detections max
        verbose=False,
    )[0]

    mask = np.zeros_like(image, dtype=np.uint16)
    if result.masks is not None:
        mask = result.masks.cpu().numpy().data[0]  # First mask only
        mask = resize(mask, image.shape, order=0) == 1
        mask[mask] = 1  # TODO: improvable??

        # Keep one of the channels only
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        # Fill-in the mask TODO: unnecessary?
        mask = ndi.binary_fill_holes(
            mask, structure=ndi.generate_binary_structure(2, 1)
        )

    if len(mask.shape) == 3:
        mask = mask[..., 0]

    return mask


def _handle_3d_predict(image, model):
    mask_3d = []
    for slice_idx, z_slice in enumerate(tqdm(image, desc="Detecting lungs")):
        mask_2d = _handle_2d_predict(z_slice, model)
        mask_3d.append(mask_2d)
    mask_3d = np.stack(mask_3d)

    # Dilate in the Z direcion to suppress missing frames
    mask_3d = ndi.binary_dilation(
        mask_3d, structure=ndi.generate_binary_structure(3, 1), iterations=2
    )

    return mask_3d


### For V2 ###


def _predict_slice(image_slice, model):
    h, w = image_slice.shape[:2]

    image_slice = np.clip(image_slice * 255, 0, 255).astype(np.uint8)

    image_slice = gray2rgb(image_slice)

    result = model.predict(
        image_slice,
        conf=0.1,
        max_det=2,
        iou=0.5,
        verbose=False,
    )[0]

    if result.masks is not None:
        masks = result.masks.data
        semantic_mask = masks.any(dim=0).cpu().numpy().astype(np.uint8)
        semantic_mask = resize(semantic_mask, (h, w), order=0)

        return semantic_mask
    else:
        return np.zeros((h, w), dtype=np.uint8)


def _predict_3d(image, model, axis=0):
    if axis != 0:
        image = np.swapaxes(image, 0, axis)

    mask3d = []
    for img_slice in image:
        mask2d = _predict_slice(img_slice, model)
        mask3d.append(mask2d)
    mask3d = np.stack(mask3d, axis=0)

    # Dilate in the Z direcion to suppress missing frames
    mask3d = ndi.binary_dilation(
        mask3d, structure=ndi.generate_binary_structure(3, 1), iterations=2
    )

    if axis != 0:
        mask3d = np.swapaxes(mask3d, 0, axis)

    return mask3d


def _handle_3d_predict_multi_axial(image, model):
    # Normalize the image
    norm_lb = image.min()
    sample = image[image != image.min()]
    norm_ub = np.percentile(sample, 96)
    image_normed = (image - norm_lb) / (norm_ub - norm_lb)

    # Predict along two axes
    mask_ax0 = _predict_3d(image_normed, model, axis=0)
    mask_ax1 = _predict_3d(image_normed, model, axis=1)

    # Logical AND #TODO or logical OR?
    mask = np.logical_and(mask_ax0, mask_ax1)

    return mask


### For V3 ###

def _handle_2d_predict_sem(image, model):
    image_normed = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    image_rgb = gray2rgb(image_normed)

    results = model.predict(
        image_rgb, 
        conf=0.1, 
        verbose=False,
    )

    semantic_mask = results[0].semantic_mask.data.cpu().numpy()
    
    return semantic_mask


def _handle_3d_predict_sem(image, model):
    mask_3d = []
    for slice_idx, z_slice in enumerate(tqdm(image, desc="Detecting lungs")):
        mask_2d = _handle_2d_predict_sem(z_slice, model)
        mask_3d.append(mask_2d)
    mask_3d = np.stack(mask_3d)

    # Dilate in the Z direcion to suppress missing frames
    mask_3d = ndi.binary_dilation(
        mask_3d, structure=ndi.generate_binary_structure(3, 1), iterations=2
    )
        
    return mask_3d
    

class LungsPredictor:
    """
    Predictor for lung segmentation using a pre-trained YOLO model.

    Parameters
    ----------
    model : str
        Identifier of the YOLO model to use. Must be a key in configuration.YOLO_MODELS.
        
    Methods
    ----------
    predict(): Segment lungs in a 3D image using the YOLO model.
    fast_predict(): Quickly segment lungs by skipping slices in Z and interpolating predictions.
    compute_3d_roi(): Compute a 3D region of interest by segmenting the lungs and cropping the image around them.
    """

    def __init__(self, model: str):
        model_path = os.path.expanduser(
            os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".lutuflow")
        )

        model_url, model_known_hash = YOLO_MODELS.get(model)

        pooch.retrieve(
            url=model_url,
            known_hash=model_known_hash,
            path=model_path,
            progressbar=True,
            fname=f"lutuflow-{model}.pt",
        )

        self.model_name = model

        self.model = YOLO(os.path.join(model_path, f"lutuflow-{model}.pt"))

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Segment lungs in a 3D image using the YOLO model.

        Parameters
        ----------
        image : np.ndarray
            3D image array.

        Returns
        -------
        np.ndarray
            Binary mask of the lung region with same shape as input image.
        """
        if self.model_name == "image-to-lungs":
            mask_3d = _handle_3d_predict(image, self.model)
        elif self.model_name == "v2":
            mask_3d = _handle_3d_predict_multi_axial(image, self.model)
        elif self.model_name == "roi-to-lungs":
            mask_3d = _handle_3d_predict_sem(image, self.model)

        mask_3d = _keep_biggest_object(mask_3d)
        
        return mask_3d

    def fast_predict(self, image: np.ndarray, skip_level: int = 1) -> np.ndarray:
        """
        Quickly segment lungs by skipping slices in Z and interpolating predictions.

        Parameters
        ----------
        image : np.ndarray
            3D image array.
        skip_level : int, optional
            Frame skip interval for faster prediction (default is 1).

        Returns
        -------
        np.ndarray
            Binary mask of the lung region with same shape as input image.
        """
        rz, ry, rx = image.shape
        mask = np.zeros(image.shape, dtype=np.uint8)
        image_partial = image[::skip_level]
        mask_partial = self.predict(image_partial)
        mask[::skip_level] = mask_partial
        range_z = np.arange(rz)
        annotated_slices = range_z[::skip_level]
        for y in range(ry):
            for x in range(rx):
                values = mask_partial[:, y, x]
                interp_func = interp1d(
                    annotated_slices,
                    values,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=0,
                )
                mask[:, y, x] = interp_func(range_z)

        return mask

    def compute_3d_roi(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a 3D region of interest by segmenting the lungs and cropping the image around them.

        Parameters
        ----------
        image : np.ndarray
            3D image array.

        Returns
        -------
        roi : np.ndarray
            Cropped image array focused on the lungs.
        roi_mask : np.ndarray
            Binary mask of the lungs within the cropped image.
        """
        mask = self.predict(image)

        roi, roi_mask = _extract_3d_roi(image, mask)

        return roi, roi_mask

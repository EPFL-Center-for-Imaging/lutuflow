import numpy as np
import scipy.ndimage as ndi
from skimage.measure import marching_cubes
from skimage.filters import gaussian
from vedo import Points


def _apply_transform(image, Phi, order: int = 3):
    """Applies an affine transformation to warp a 3D image."""
    warped = ndi.affine_transform(
        image, matrix=Phi[:3, :3], offset=Phi[:3, 3], order=order
    )

    return warped


def _fit_affine_from_lungs_masks(lungs0: np.ndarray, lungs1: np.ndarray):
    """Estimates the affine transformation that brings lung1 onto lung0."""
    verts0, *_ = marching_cubes(gaussian(lungs0.astype(float), sigma=1), level=0.5)
    verts1, *_ = marching_cubes(gaussian(lungs1.astype(float), sigma=1), level=0.5)

    aligned_pts1 = (
        Points(verts1).clone().align_to(Points(verts0), invert=True, use_centroids=True)
    )

    Phi = aligned_pts1.transform.matrix

    return Phi


def register_tumor_series(
    tumor_timeseries: np.ndarray,
    lungs_timeseries: np.ndarray,
):
    """
    Register a 4D array (TZYX) of tumor segmentation masks using the lungs segmentation masks.

    Parameters
    ----------
    tumor_timeseries : np.ndarray
        4D array (TZYX) of tumor segmentation masks.
    lungs_timeseries : np.ndarray
        4D array of lung masks for registration.
        
    Returns
    -------
    registered_tumor_timeseries : np.ndarray
        4D array (TZYX) of tumor segmentation masks registered to the first frame.
    """
    tumors0 = tumor_timeseries[0]
    registered_tumor_timeseries = np.empty_like(tumor_timeseries)
    registered_tumor_timeseries[0] = tumors0

    lung0 = lungs_timeseries[0]

    for k, (tumors1, lungs1) in enumerate(
        zip(tumor_timeseries[1:], lungs_timeseries[1:])
    ):
        Phi = _fit_affine_from_lungs_masks(lung0, lungs1)

        warped_tumors1 = _apply_transform(tumors1, Phi, order=0)
        registered_tumor_timeseries[k + 1] = warped_tumors1

    return registered_tumor_timeseries

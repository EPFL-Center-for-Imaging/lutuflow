from pathlib import Path

import imaging_server_kit as sk
import numpy as np
from lutuflow.core import (
    NNUNET_MODELS,
    LungsPredictor,
    TumorPredictor,
    generate_tracked_tumors,
    run_tracking,
)
from lutuflow.core.track import (
    combine_images,
    regenerate_linkage_df,
    to_formatted_df,
)

ROOT = str(Path.home().resolve())


@sk.algorithm(
    name="5. Save tracking table",
    description="Save a tracking table as a CSV file.",
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    parameters={
        "tracked_tumors": sk.Mask(
            name="Tracked tumors",
            description="Tracked tumor time series segmentation mask (TZYX).",
            dimensionality=[4],
        ),
        "untracked_tumors": sk.Mask(
            name="Untracked tumors",
            description="Untracked tumor time series segmentation mask (TZYX).",
            dimensionality=[4],
        ),
        "save_dir": sk.String(
            name="Directory",
            description="Directory where to save the CSV file.",
            default=ROOT,
        ),
        "file_name": sk.String(
            name="CSV File name",
            description="CSV file name.",
            default="tracking_table.csv",
        ),
    },
    tileable=False,
)
def sk_save_csv(tracked_tumors, untracked_tumors, save_dir, file_name):
    if not file_name.endswith(".csv"):
        file_name = file_name + ".csv"

    csv_file = save_dir / file_name

    linkage_df = regenerate_linkage_df(
        tracked_tumor_series=tracked_tumors,
        untracked_tumor_series=untracked_tumors,
    )

    formatted_df = to_formatted_df(linkage_df)

    formatted_df.to_csv(csv_file)

    return sk.Notification(f"Saved {csv_file}")


@sk.algorithm(
    name="4. Track tumors",
    description="Track lung tumors in series of 3D CT scans.",
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tags=["Tracking"],
    parameters={
        "tumor_series": sk.Mask(
            name="Tumor mask series (TXYX)",
            description="4D (TZYX) segmentation mask of the tumors (untracked).",
            dimensionality=[4],
        ),
        "image_series": sk.Image(
            name="CT scan series (TZYX)",
            description="4D (TZYX) CT image of the lung ROIs (used to register the tumors before tracking).",
            required=False,
            dimensionality=[4],
        ),
        "max_dist_px": sk.Integer(
            name="Max distance (px)",
            description="Distance cutoff (in pixels) for object tracking between consecutive frames.",
            min=0,
            default=30,
        ),
        "max_volume_diff_rel": sk.Float(
            name="Max volume change",
            description="Maximum relative volume change between two consecutive scans. 1.0 means a tumor can at most double in size or shring by half of its size between two consecutive scans.",
            default=1.0,
            min=0.0,
        ),
        "dist_weight_ratio": sk.Float(
            name="Distance/Volume importance",
            description="Relative importance given to the distance between tumors and their similarity in size for the tracking algorithm. Decrease this value to favour matches based on volume similarity.",
            min=0,
            max=1,
            step=0.05,
            default=0.9,
        ),
    },
    tileable=False,
)
def sk_tracking(
    tumor_series,
    image_series,
    max_dist_px,
    max_volume_diff_rel,
    dist_weight_ratio,
):
    linkage_df = run_tracking(
        tumor_timeseries=tumor_series,
        image_timeseries=image_series,
        max_dist_px=max_dist_px,
        dist_weight_ratio=dist_weight_ratio,
        max_volume_diff_rel=max_volume_diff_rel,
        memory=0,
        skip_level=8,
        remove_partially_tracked=False,
    )

    tracked_labels_timeseries = generate_tracked_tumors(tumor_series, linkage_df)

    n_tracked_tumors = len(linkage_df["tumor"].unique().tolist())

    if n_tracked_tumors == 0:
        return sk.Notification("No tumors tracked")
    elif tracked_labels_timeseries is None:
        return sk.Notification("Tracked tumors are None", meta={"level": "warning"})

    return (
        sk.Mask(tracked_labels_timeseries, name="Tracked tumors"),
        f"{n_tracked_tumors} tumors tracked",
    )


@sk.algorithm(
    name="1. Crop original scans",
    description="Automatically crop 3D CT scans around the lungs region.",
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tags=["Segmentation"],
    parameters={
        "image": sk.Image(
            name="CT scan image (3D)",
            description="Raw 3D CT scan image (ZYX).",
            dimensionality=[3],
        ),
        "return_lungs": sk.Bool(
            name="Return lungs mask",
            default=False,
            description="Set this value to True to return a segmentation mask of the lungs along with the image ROI.",
        ),
    },
    tileable=False,
)
def sk_lungs_seg(image, return_lungs):
    predictor = LungsPredictor(model="image-to-lungs")
    lungs_roi, lungs_mask = predictor.compute_3d_roi(image)
    returns = [sk.Image(lungs_roi, name="Image (ROI)")]
    if return_lungs:
        returns.append(sk.Mask(lungs_mask, name="Lungs (ROI)"))
    return returns


@sk.algorithm(
    name="2. Segment tumors",
    description="Segment tumors in 3D mice CT scans.",
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tags=["Segmentation"],
    parameters={
        "image": sk.Image(
            name="CT scan image (3D)",
            description="CT scan image (3D, ZYX) or series (3D+t, TZYX) of the lung region of interest (ROI).",
            dimensionality=[3, 4],
        ),
        "model": sk.Choice(
            name="Model",
            description="Model to use for the tumor detection.",
            items=list(NNUNET_MODELS.keys()),
        ),
    },
    tileable=False,
)
def sk_tumor_seg(image, model):
    predictor = TumorPredictor(model)
    if image.ndim == 3:
        mask = predictor.predict(image)
        return sk.Mask(mask)
    elif image.ndim == 4:
        mask = np.zeros(image.shape, dtype=np.uint16)
        for k, scan in enumerate(image):  # Assuming the first dimension is T
            mask[k] = predictor.predict(scan)
            yield sk.Mask(mask)  # Progressively output the segmentation masks
    else:
        return sk.Notification(
            f"Images of dimensionality {image.ndim} are not supported.",
            meta={"level": "warning"},
        )


# Combining 2 to 4 images
@sk.algorithm(
    name="3. Combine 2 CT scans",
    description="Combine 3D images (ZYX) into a single 4D image where the first axis is time (TZYX).",
    parameters={
        "image0": sk.Image(name="Image 1"),
        "image1": sk.Image(name="Image 2"),
    },
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tileable=False,
)
def sk_combine_image_pair(image0, image1):
    # TODO: something's not working with sk here.
    image_series = combine_images([image0, image1])
    return sk.Image(image_series, name="Combined")


@sk.algorithm(
    name="3. Combine 3 CT scans",
    description="Combine 3D images (ZYX) into a single 4D image where the first axis is time (TZYX).",
    parameters={
        "image0": sk.Image(name="Image 1"),
        "image1": sk.Image(name="Image 2"),
        "image2": sk.Image(name="Image 3"),
    },
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tileable=False,
)
def sk_combine_image_trio(image0, image1, image2):
    image_series = combine_images([image0, image1, image2])
    return sk.Image(image_series, name="Combined")


@sk.algorithm(
    name="3. Combine 4 CT scans",
    description="Combine 3D images (ZYX) into a single 4D image where the first axis is time (TZYX).",
    parameters={
        "image0": sk.Image(name="Image 1"),
        "image1": sk.Image(name="Image 2"),
        "image2": sk.Image(name="Image 3"),
        "image3": sk.Image(name="Image 4"),
    },
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tileable=False,
)
def sk_combine_image_quatuor(image0, image1, image2, image3):
    image_series = combine_images([image0, image1, image2, image3])
    return sk.Image(image_series, name="Combined")


# Combining 2 to 4 masks
@sk.algorithm(
    name="3. Combine 2 tumor masks",
    description="Combine 3D masks (ZYX) into a single 4D mask where the first axis is time (TZYX).",
    parameters={
        "mask0": sk.Mask(name="Mask 1"),
        "mask1": sk.Mask(name="Mask 2"),
    },
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tileable=False,
)
def sk_combine_mask_pair(mask0, mask1):
    mask_series = combine_images([mask0, mask1])
    return sk.Mask(mask_series, name="Combined")


@sk.algorithm(
    name="3. Combine 3 tumor masks",
    description="Combine 3D masks (ZYX) into a single 4D mask where the first axis is time (TZYX).",
    parameters={
        "mask0": sk.Mask(name="Mask 1"),
        "mask1": sk.Mask(name="Mask 2"),
        "mask2": sk.Mask(name="Mask 3"),
    },
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tileable=False,
)
def sk_combine_mask_trio(mask0, mask1, mask2):
    mask_series = combine_images([mask0, mask1, mask2])
    return sk.Mask(mask_series, name="Combined")


@sk.algorithm(
    name="3. Combine 4 tumor masks",
    description="Combine 3D masks (ZYX) into a single 4D mask where the first axis is time (TZYX).",
    parameters={
        "mask0": sk.Mask(name="Mask 1"),
        "mask1": sk.Mask(name="Mask 2"),
        "mask2": sk.Mask(name="Mask 3"),
        "mask3": sk.Mask(name="Mask 4"),
    },
    project_url="https://github.com/EPFL-Center-for-Imaging/lutuflow",
    tileable=False,
)
def sk_combine_mask_quatuor(mask0, mask1, mask2, mask3):
    mask_series = combine_images([mask0, mask1, mask2, mask3])
    return sk.Mask(mask_series, name="Combined")


multikit = sk.combine(
    [
        sk_lungs_seg,
        sk_tumor_seg,
        sk_tracking,
        sk_combine_image_pair,
        sk_combine_image_trio,
        sk_combine_image_quatuor,
        sk_combine_mask_pair,
        sk_combine_mask_trio,
        sk_combine_mask_quatuor,
        sk_save_csv,
    ],
    name="LuTuFlow",
)

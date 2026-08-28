import os
from pathlib import Path
import numpy as np
import skimage.io

from .kit import (
    sk_tracking,
    sk_lungs_seg,
    sk_tumor_seg,
    sk_combine_image_pair,
    sk_combine_image_trio,
    sk_combine_image_quatuor,
    sk_save_csv,
)


def predict(image_file, out_dir, model):
    """CLI functionality for segmenting tumors"""
    if out_dir is None:
        out_dir = Path(image_file).parent
    else:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            os.makedirs(out_dir)

    image_name = Path(image_file).stem

    out_file = out_dir / f"{image_name}_tumors.tif"

    image = skimage.io.imread(image_file)

    # This can be a mask or a notification message
    result = sk_tumor_seg(image=image, model=model)

    if isinstance(result, np.ndarray):
        skimage.io.imsave(out_file, result)
        print(f"✅ Saved {out_file}")
    else:
        print(result)


def crop(image_file, out_dir, with_lungs):
    """CLI functionality for cropping the image and segmenting the lungs"""
    if out_dir is None:
        out_dir = Path(image_file).parent
    else:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            os.makedirs(out_dir)

    image_name = Path(image_file).stem

    out_file_roi = out_dir / f"{image_name}_roi.tif"
    out_file_lungs = out_dir / f"{image_name}_lungs.tif"

    image = skimage.io.imread(image_file)

    result = sk_lungs_seg(image=image, return_lungs=with_lungs)

    image_roi = result[0]
    skimage.io.imsave(out_file_roi, image_roi)
    print(f"✅ Saved {out_file_roi}")

    if with_lungs:
        lungs_roi = result[1]
        skimage.io.imsave(out_file_lungs, lungs_roi)
        print(f"✅ Saved {out_file_lungs}")


def combine(*files, out_dir):
    """CLI functionality for creating time series from several image files"""
    n_files = len(files)

    if n_files < 2:
        print(f"⚠️ Please provide at least 2 files to combine (Got {n_files}).")
        return

    elif n_files > 4:
        print(f"⚠️ Please provide at most 4 files to combine (Got {n_files}).")
        return

    if out_dir is None:
        out_dir = Path(files[0]).parent
    else:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            os.makedirs(out_dir)

    image_name = Path(files[0]).stem  # good choice?

    out_file = out_dir / f"{image_name}_{n_files:02d}-series.tif"

    images = [skimage.io.imread(image_file) for image_file in files]

    if n_files == 2:
        image_series = sk_combine_image_pair(*images)
    elif n_files == 3:
        image_series = sk_combine_image_trio(*images)
    elif n_files == 4:
        image_series = sk_combine_image_quatuor(*images)

    skimage.io.imsave(out_file, image_series)
    print(f"✅ Saved {out_file}")


def track(
    out_dir,
    labels_file,
    image_file,
    max_dist_px,
    dist_weight_ratio,
    max_volume_diff_rel,
):
    """CLI functionality for tracking tumors"""
    labels_timeseries = skimage.io.imread(labels_file)
    if image_file is not None:
        image_timeseries = skimage.io.imread(image_file)
    else:
        image_timeseries = None

    if out_dir is None:
        out_dir = Path(labels_file).parent
    else:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            os.makedirs(out_dir)

    image_name = Path(labels_file).stem

    tracked_labels_file = out_dir / f"{image_name}_tracked.tif"

    result = sk_tracking(
        tumor_series=labels_timeseries,
        image_series=image_timeseries,
        max_dist_px=max_dist_px,
        max_volume_diff_rel=max_volume_diff_rel,
        dist_weight_ratio=dist_weight_ratio,
    )

    # Result can be a tumor mask or a notification message
    if isinstance(result, np.ndarray):
        tracked_tumors = result[0]

        # Save the CSV
        save_result = sk_save_csv(
            tracked_tumors=tracked_tumors,
            untracked_tumors=labels_timeseries,
            save_dir=out_dir,
            file_name=f"{image_name}_tracks.csv",
        )
        print(save_result)

        # Save the tumors too
        skimage.io.imsave(tracked_labels_file, tracked_tumors)
        print(f"✅ Saved {tracked_labels_file}")
    else:
        print(result)

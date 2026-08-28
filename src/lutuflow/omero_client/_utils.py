from pathlib import Path
from typing import Union
import numpy as np
import skimage.io
from lutuflow.omero_client._context import ImageContext
from lutuflow.omero_client._tags_processor import TagsProcessor


def load_ct_from_folder(image_dir: Union[str, Path]) -> ImageContext:
    """Creates a 3D TIFF from a series of 2D tiffs in a folder, according to lab conventions."""
    tiff_files = Path(image_dir).glob("*.tif")
    # Remove the file with "rec_spr" in the name which is an overview of the mouse
    tiff_image_files = sorted(
        [file for file in tiff_files if "rec_spr" not in Path(file).stem.split("~")[-1]]
    )

    # Assuming that all files follow the naming convention, simply check the pattern in the first file
    exp_name, scan_time, specimen = (
        Path(tiff_image_files[0]).stem.split("~")[0].split("_")
    )

    # Check the naming conventions
    if TagsProcessor.get_specimen_tags([specimen]) is None:
        raise ValueError(
            f"Name {specimen} does not comply with specimen naming convention (C***** or Animal*****)."
        )

    if not len(TagsProcessor.get_scan_time_tags([scan_time])) == 1:
        raise ValueError(
            f"Name {scan_time} does not comply with scan time naming convention (SCAN* or T*)."
        )

    time_idx = TagsProcessor.get_scan_time_idx(scan_time)

    # Read files into a 3D tiff
    image = np.array([skimage.io.imread(file) for file in tiff_image_files])

    image_ctx = ImageContext(
        image_class="image",
        image=image,
        time_tag=scan_time,
        time_idx=time_idx,
        specimen_tag=specimen,
    )

    return image_ctx

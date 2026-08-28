from itertools import product
from typing import List, Optional
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.util import map_array
from laptrack import LapTrack

from lutuflow.core.register import register_tumor_series
from lutuflow.core.lungs import LungsPredictor


def combine_images(image_array_list: List[np.ndarray]) -> np.ndarray:
    """
    Combine a list of 3D images with varying shapes into a 4D time series array.

    Parameters
    ----------
    image_array_list : list of np.ndarray
        List of 3D image arrays of shape (Z, Y, X).

    Returns
    -------
    np.ndarray
        4D array of shape (T, Z_max, Y_max, X_max) where each image is centered and padded to the maximum spatial dimensions.
    """
    n_images = len(image_array_list)
    image_type = image_array_list[0].dtype
    image_shapes = np.stack([np.array(img.shape) for img in image_array_list])
    output_shape = [n_images]
    output_shape.extend(list(np.max(image_shapes, axis=0)))

    timeseries = np.empty(output_shape, dtype=image_type)
    for k, (image, image_shape) in enumerate(zip(image_array_list, image_shapes)):
        delta = (output_shape[1:] - image_shape) // 2
        timeseries[k][
            delta[0] : delta[0] + image_shape[0],
            delta[1] : delta[1] + image.shape[1],
            delta[2] : delta[2] + image.shape[2],
        ] = image

    return timeseries


def initialize_df(labels_timeseries: np.ndarray, properties: List[str]) -> pd.DataFrame:
    """
    Initialize a DataFrame of region properties for each frame in a time series of labeled images.

    Parameters
    ----------
    labels_timeseries : np.ndarray
        4D array (TZYX) of labeled segmentation masks over time.
    properties : list of str
        List of regionprops properties to extract (e.g., ["area", "centroid"]).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the specified properties and frame information for each labeled region.
    """
    dfs = []
    for t, frame in enumerate(labels_timeseries):
        if frame.sum() == 0:
            continue
        df = pd.DataFrame(regionprops_table(frame, properties=properties))
        df["frame_forward"] = t
        dfs.append(df)
    coordinate_df = pd.concat(dfs)
    # Invert the frame IDs to be able to track particles from the end
    coordinate_df["frame"] = (
        coordinate_df["frame_forward"].max() - coordinate_df["frame_forward"]
    )

    return coordinate_df


def _track_laptrack(
    df: pd.DataFrame,
    labels_timeseries: np.ndarray,
    max_dist_px: int,
    memory: int,
    dist_weight_ratio: float,
    max_volume_diff_rel: float,
):
    """Tracking with Laptrack."""
    n_frames = len(labels_timeseries)
    records = []
    for frame_idx in range(n_frames - 1):
        unique_labels_t0 = df.loc[df["frame"] == frame_idx, "label"].unique().astype(int)
        unique_labels_t1 = df.loc[df["frame"] == frame_idx + 1, "label"].unique().astype(int)
        for l1, l2 in product(unique_labels_t0, unique_labels_t1):
            # Ignore the background
            if (l1 == 0) | (l2 == 0):
                continue

            sub_df1 = df.loc[
                (df["label"] == l1) & (df["frame"] == frame_idx), ["volume", "z", "y", "x"]
            ]
            sub_df2 = df.loc[
                (df["label"] == l2) & (df["frame"] == frame_idx + 1), ["volume", "z", "y", "x"]
            ]

            if len(sub_df1) & len(sub_df2):
                # Compute relative volume difference for this labels pair
                volume_l1 = sub_df1["volume"].values[0]
                volume_l2 = sub_df2["volume"].values[0]
                volume_diff_l1_l2 = np.abs(volume_l2 - volume_l1) / max(
                    volume_l1, volume_l2
                )

                # Compute euclidean distance
                coord_l1 = sub_df1[["z", "y", "x"]].values[0]
                coord_l2 = sub_df2[["z", "y", "x"]].values[0]
                euclidean_dist_l1_l2 = np.linalg.norm(coord_l1 - coord_l2)
            else:
                volume_diff_l1_l2 = np.nan
                euclidean_dist_l1_l2 = np.nan

            records.append(
                {
                    "frame": frame_idx,
                    "label1": l1,
                    "label2": l2,
                    "volume_diff": volume_diff_l1_l2,
                    "euclidean_dist": euclidean_dist_l1_l2,
                }
            )
    
    combinatorial_df = pd.DataFrame.from_records(records)

    # Remove NaN values
    # combinatorial_df.dropna(inplace=True)
    combinatorial_df = (
        pd.DataFrame.from_records(records)
        .dropna()
        .drop_duplicates()
    )

    combinatorial_df["euclidean_dist_normalized"] = (
        combinatorial_df["euclidean_dist"] - combinatorial_df["euclidean_dist"].mean()
    ) / combinatorial_df["euclidean_dist"].std()
    
    combinatorial_df["volume_diff_normalized"] = (
        combinatorial_df["volume_diff"] - combinatorial_df["volume_diff"].mean()
    ) / combinatorial_df["volume_diff"].std()
    
    combinatorial_df = combinatorial_df.set_index(["frame", "label1", "label2"])

    def metric(c1: tuple, c2: tuple) -> float:
        (frame1, label1), (frame2, label2) = c1, c2

        if frame1 == frame2 + 1:
            tmp = (frame1, label1)
            (frame1, label1) = (frame2, label2)
            (frame2, label2) = tmp
        
        assert frame1 + 1 == frame2

        ind = (frame1, label1, label2)

        dist = combinatorial_df.loc[ind, "euclidean_dist_normalized"]
        
        vols = combinatorial_df.loc[ind,"volume_diff_normalized"]

        return float(dist) * dist_weight_ratio + float(vols) * (1 - dist_weight_ratio)

    max_dist_normalized = (
        max_dist_px - combinatorial_df["euclidean_dist"].mean()
    ) / combinatorial_df["euclidean_dist"].std()
    
    max_volume_diff_normalized = (
        max_volume_diff_rel - combinatorial_df["volume_diff"].mean()
    ) / combinatorial_df["volume_diff"].std()
    
    max_metric_cutoff = (
        max_dist_normalized * dist_weight_ratio
        + max_volume_diff_normalized * (1 - dist_weight_ratio)
    )

    lt = LapTrack(
        track_dist_metric=metric,  # custom metric
        gap_closing_max_frame_count=memory,  # "memory" parameter
        track_cost_cutoff=max_metric_cutoff,  # Maximum difference criterion between two linkages
        splitting_cost_cutoff=False,  # non-splitting case
        merging_cost_cutoff=False,  # non-merging case
    )

    df_without_bg = df[df["label"] != 0].copy()

    linkage_df, *_ = lt.predict_dataframe(df_without_bg, coordinate_cols=["frame", "label"], only_coordinate_cols=False)

    linkage_df.reset_index(inplace=True)
    linkage_df.rename(columns={"tree_id": "tumor", "frame_forward": "scan"}, inplace=True)
    linkage_df.drop(["frame", "track_id", "index"], axis="columns", inplace=True)

    linkage_df["tumor"] = linkage_df["tumor"] + 1

    return linkage_df


def generate_tracked_tumors(labels_timeseries: np.ndarray, linkage_df: pd.DataFrame) -> np.ndarray:
    """
    Generate a time series of segmentation masks with consistent labels based on tracking results.

    Parameters
    ----------
    labels_timeseries : np.ndarray
        4D array (TZYX) of original labeled segmentation masks.
    linkage_df : pd.DataFrame
        DataFrame linking original labels to tracked object IDs with columns ["scan", "tumor", "label"].

    Returns
    -------
    np.ndarray
        4D array of segmentation masks where labels are replaced by tracked tumor IDs.
    """
    corrected_timeseries = np.zeros_like(labels_timeseries)

    unique_times = linkage_df["scan"].unique()
    for t in unique_times:
        scan_idx = int(t)
        dft = linkage_df[linkage_df["scan"] == t][["tumor", "label"]]
        new_labels = dft["tumor"].values.astype(corrected_timeseries.dtype)
        old_labels = dft["label"].values.astype(corrected_timeseries.dtype)
        # Ignore the background labels
        new_labels_filtered = new_labels[(old_labels != 0) & (new_labels != 0)]
        old_labels_filtered = old_labels[(old_labels != 0) & (new_labels != 0)]
        map_array(
            labels_timeseries[scan_idx],
            old_labels_filtered,  # The values to map from.
            new_labels_filtered,  # The values to map to.
            out=corrected_timeseries[scan_idx],
        )

    return corrected_timeseries


def run_tracking(
    tumor_timeseries: np.ndarray,
    lungs_timeseries: Optional[np.ndarray] = None,
    image_timeseries: Optional[np.ndarray] = None,
    max_dist_px=30,
    dist_weight_ratio=0.9,
    max_volume_diff_rel=1.0,
    memory=0,
    skip_level=8,
    remove_partially_tracked: bool = False,
) -> pd.DataFrame:
    """
    Track objects across a time series of labeled segmentation masks.

    Parameters
    ----------
    tumor_timeseries : np.ndarray
        4D array of tumor segmentation masks (TZYX) to track.
    lungs_timeseries : np.ndarray, optional
        4D array of lungs segmentation masks.
    image_timeseries : np.ndarray, optional
        4D array of original images (CT scans).
    max_dist_px : int, default 30
        Maximum allowed movement (in pixels) between frames for linking.
    max_volume_diff_rel : float, default 1.0
        Maximum relative volume change for linking in LapTrack metric.
    dist_weight_ratio : float, default 0.9
        Weight ratio between spatial distance and volume difference for LapTrack metric.
    memory : int, default 0
        Number of frames to allow objects to disappear and reappear.
    skip_level : int, default 8
        Frame skipping interval for lung-based registration.
    remove_partially_tracked : bool, default True
        If True, remove tracked objects that do not appear in all frames.

    Returns
    -------
    pd.DataFrame
        DataFrame containing tracked object linkages with columns ['tumor', 'scan', 'label', ...].
    """
    # Volumes are computed on the original labels
    df_original_labels = initialize_df(tumor_timeseries, properties=["area", "label"])

    n_frames = len(tumor_timeseries)

    if n_frames == 1:
        registered_labels_timeseries = tumor_timeseries
    else:
        if lungs_timeseries is None:
            if image_timeseries is None:
                raise RuntimeError("No images or lungs timeseries provided for the registration.")
            
            predictor = LungsPredictor(model="roi-to-lungs")
            
            lungs_timeseries = np.array(
                [
                    predictor.fast_predict(frame, skip_level=skip_level)
                    for frame in image_timeseries
                ]
            )
            
        registered_labels_timeseries = register_tumor_series(
            tumor_timeseries=tumor_timeseries,
            lungs_timeseries=lungs_timeseries,
        )

    # Positions are computed on the registered labels
    df_registered_labels = initialize_df(
        registered_labels_timeseries,
        properties=["centroid", "label"],
    )

    df = pd.merge(
        df_original_labels, df_registered_labels, on=["label", "frame_forward", "frame"]
    )

    df.rename(
        columns={
            "centroid-0": "z",
            "centroid-1": "y",
            "centroid-2": "x",
            "area": "volume",
        },
        inplace=True,
    )
    
    linkage_df = _track_laptrack(
        df,
        registered_labels_timeseries,
        max_dist_px,
        memory,
        dist_weight_ratio,
        max_volume_diff_rel,
    )
    
    linkage_df = linkage_df.merge(
        pd.DataFrame({"length": linkage_df["tumor"].value_counts()}),
        left_on="tumor",
        right_index=True,
    )

    if remove_partially_tracked:
        # Remove tracks (and objects) that don't appear in every frame
        linkage_df = linkage_df[linkage_df["length"] == len(tumor_timeseries)]

    linkage_df = linkage_df.sort_values(by="length", ascending=False)
    linkage_df = linkage_df.fillna(0)

    if len(linkage_df) == 0:
        print("⚠️ No tumors were tracked.")

    return linkage_df


def regenerate_linkage_df(
    tracked_tumor_series: np.ndarray, untracked_tumor_series: np.ndarray
) -> pd.DataFrame:
    """
    Recompute the linkage DataFrame mapping original labels to tracked tumor IDs.

    Parameters
    ----------
    tracked_tumor_series : np.ndarray
        4D array of tracked segmentation masks (time, z, y, x) with tumor IDs as labels.
    untracked_tumor_series : np.ndarray
        4D array of original labeled segmentation masks.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['scan', 'label', 'tumor', 'volume'] indicating mapping between frames, labels, and tumors.
    """
    df_original_labels = initialize_df(untracked_tumor_series, properties=["area", "label"])

    rows = []
    for frame_forward, (untracked_labels_frame, tracked_labels_frame) in enumerate(
        zip(untracked_tumor_series, tracked_tumor_series)
    ):
        for label_idx in np.unique(untracked_labels_frame):
            if label_idx == 0:  # Ignore the background
                continue

            uniques = np.unique(
                tracked_labels_frame[untracked_labels_frame == label_idx]
            )
            if len(uniques) != 1:
                raise ValueError("Unexpected multiple unique labels found in the mask.")
            tumor_idx = uniques[0]

            # The tumor was dropped during tracking
            if tumor_idx == 0:
                continue

            filt = (df_original_labels["label"] == label_idx) & (
                df_original_labels["frame_forward"] == frame_forward
            )
            tumor_volume = df_original_labels.loc[filt, "area"]
            tumor_volume = tumor_volume.values[0]

            row = {
                "scan": frame_forward,
                "label": label_idx,
                "tumor": tumor_idx,
                "volume": tumor_volume,
            }
            rows.append(row)

    linkage_df = pd.DataFrame(rows)

    return linkage_df


def to_formatted_df(linkage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot and format the linkage DataFrame with volumes and labels across scans.

    Parameters
    ----------
    linkage_df : pd.DataFrame
        DataFrame containing tracked object linkages with columns ['tumor', 'scan', 'volume', 'label'].

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with tumor IDs as rows and scan-specific columns for volume and label, including fold-change calculations.
    """
    grouped_df = linkage_df.groupby(["tumor", "scan"]).mean()
    grouped_df_index_reset = grouped_df.reset_index()[
        ["tumor", "scan", "volume", "label"]
    ]
    formatted_df = grouped_df_index_reset.pivot(
        index="tumor", columns="scan", values=["volume", "label"]
    ).reset_index()
    formatted_df.columns = ["Tumor ID"] + [
        f"{i} - SCAN{scan_id:02d}" for (i, scan_id) in formatted_df.columns[1:]
    ]

    volume_columns = [col for col in formatted_df.columns if "volume" in col]
    label_columns = [col for col in formatted_df.columns if "label" in col]

    if len(volume_columns):
        # Compute fold change
        initial_volume_col = volume_columns[0]
        for k, volume_col in enumerate(volume_columns[1:]):
            formatted_df[f"fold change - SCAN01 to SCAN{(k+2):02d}"] = (
                formatted_df[volume_col] - formatted_df[initial_volume_col]
            ) / formatted_df[initial_volume_col]

    # Re-order the columns
    columns_order = ["Tumor ID"]
    for volume_col, label_col in zip(volume_columns, label_columns):
        columns_order.append(label_col)
        columns_order.append(volume_col)
    fold_change_columns = [col for col in formatted_df.columns if "fold" in col]
    for fold_col in fold_change_columns:
        columns_order.append(fold_col)
    formatted_df = formatted_df[columns_order]

    # Objects that aren't tracked the whole time series get assigned label=0, which is easily identifyable in downstream processing
    formatted_df = formatted_df.fillna(0)

    return formatted_df


def to_linkage_df(formatted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a formatted pivoted DataFrame back into a linkage DataFrame format.

    Parameters
    ----------
    formatted_df : pd.DataFrame
        Formatted DataFrame as produced by to_formatted_df, containing columns for Tumor ID and scan-specific label columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['tumor', 'scan', 'label'] mapping tumor IDs to labels in each scan.
    """
    linkage_df = pd.DataFrame(columns=["tumor", "scan", "label"])
    i = 0
    for _, row in formatted_df.iterrows():
        tumor_id = row["Tumor ID"]
        for col in formatted_df.columns:
            if col.startswith("label - SCAN"):
                scan_id = int(col.split("SCAN")[1])
                label_idx = row[col]
                linkage_df.loc[i] = [tumor_id, scan_id, label_idx]
                i = i + 1

    return linkage_df  # Note that this linkage doesn't have volumes.

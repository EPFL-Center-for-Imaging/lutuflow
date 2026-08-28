from ._version import version as __version__

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core import (
    TumorPredictor,
    LungsPredictor,
    combine_images,
    run_tracking,
    regenerate_linkage_df,
    to_formatted_df,
    to_linkage_df,
    generate_tracked_tumors,
    initialize_df,
    NNUNET_MODELS,
    YOLO_MODELS,
    multikit,
)

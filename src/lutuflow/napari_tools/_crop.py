import numpy as np
import imaging_server_kit as sk
from qtpy.QtWidgets import QWidget

@sk.algorithm(
    name="Manual cropping",
    description="Interactively crop a 3D image by setting limits along the X, Y and Z axes.",
    parameters={
        "image": sk.Image(name="Image (3D)", dimensionality=[3]),
        "min_x": sk.Integer(name="Crop +X", description="Pixels to remove in X, from the start", min=0, step=5),
        "max_x": sk.Integer(name="Crop -X", description="Pixels to remove in X, from the end", step=5, min=1, default=1),
        "min_y": sk.Integer(name="Crop +Y", description="Pixels to remove in Y, from the start", min=0, step=5),
        "max_y": sk.Integer(name="Crop -Y", description="Pixels to remove in Y, from the end", step=5, min=1, default=1),
        "min_z": sk.Integer(name="Crop +Z", description="Pixels to remove in Z, from the start", min=0, step=5),
        "max_z": sk.Integer(name="Crop -Z", description="Pixels to remove in Z, from the end", step=5, min=1, default=1),
    },
    tileable=False,
)
def sk_crop(
    image, min_x: int, max_x: int, min_y: int, max_y: int, min_z: int, max_z: int
):
    crop = image[min_z:-max_z, min_y:-max_y, min_x:-max_x]
    roi_name = f"ROI_z{min_z:02d}-{max_z:02d}_y{min_y:02d}-{max_y:02d}_x{min_x:02d}-{max_x:02d}"
    return sk.Image(crop, name=roi_name, meta={"contrast_limits": [np.min(crop), np.max(crop)]}), f"ROI shape: {crop.shape}"

class ManualCropWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        widget = sk.to_qwidget(sk_crop, napari_viewer)
        self.setLayout(widget.layout())
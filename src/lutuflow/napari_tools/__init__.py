from qtpy.QtWidgets import QWidget

from ._picker import ColorPickerWidget
from ._crop import ManualCropWidget
from ._tracks import ToTracksWidget
from ._table import TableWidget
from ._remove import RemoveObjectsWidget
from ._widget import OMEROWidget

import imaging_server_kit as sk

from lutuflow.core import multikit


class PluginWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        widget = sk.to_qwidget(multikit, viewer)
        self.setLayout(widget.layout())

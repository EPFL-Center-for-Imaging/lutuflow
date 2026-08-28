import numpy as np
from napari.layers import Labels
from napari.utils import DirectLabelColormap
from napari_toolkit.widgets import setup_colorpicker
from PyQt6.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ColorPickerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        layout = QGridLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.cb_update_colormap = QComboBox()
        layout.addWidget(QLabel("Change direct color", self), 0, 0)
        layout.addWidget(self.cb_update_colormap, 0, 1)

        colorpicker_layout = QVBoxLayout()
        colorpicker = QWidget(self)
        colorpicker.setLayout(colorpicker_layout)
        self.colorpicker_widget = setup_colorpicker(
            layout=colorpicker_layout,
            initial_color=(0, 255, 0), # type: ignore
            function=self._update_direct_colormap,
        )
        layout.addWidget(colorpicker, 0, 2)

        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

    def _on_layer_change(self, e):
        self.cb_update_colormap.clear()
        for x in self.viewer.layers:
            if isinstance(x, Labels):
                self.cb_update_colormap.addItem(x.name, x.data)

    def _update_direct_colormap(self, *args, **kwargs):
        labels_data = self.cb_update_colormap.currentData()
        if labels_data is None:
            return

        labels_layer = self.viewer.layers[self.cb_update_colormap.currentText()]
        rgba = np.array(list(self.colorpicker_widget.get_color()) + [255]) / 255
        color_dict = {}
        for idx in np.unique(labels_data):
            color_dict[idx] = rgba
        color_dict[0] = np.array([0, 0, 0, 0])
        color_dict[None] = np.array([0, 0, 0, 0])
        labels_layer.colormap = DirectLabelColormap(color_dict=color_dict)
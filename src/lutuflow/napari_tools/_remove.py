from PyQt6.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, 
    QLabel, 
    QHBoxLayout,
    QCheckBox,
) 
import napari.layers
from napari.utils.events import Event


class RemoveObjectsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.active_layer = self.viewer.layers.selection.active
        
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)
        self.layout().addWidget(QLabel("Remove objects", self))
        self.cb = QCheckBox()
        self.layout().addWidget(self.cb)
        self.cb.stateChanged.connect(self._on_tick_change)
        self.viewer.layers.selection.events.changed.connect(
            self._on_layer_selection_change
        )
        self._on_layer_selection_change(self.active_layer)

    def _on_layer_selection_change(self, *args, **kwargs):
        for layer in self.viewer.layers:
            if self._delete_action in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.remove(self._delete_action)

        self.active_layer = self.viewer.layers.selection.active
        if (self.active_layer is not None) & self.cb.isChecked():
            if self._delete_action not in self.active_layer.mouse_drag_callbacks:
                if isinstance(self.active_layer, napari.layers.Labels):
                    self.active_layer.mouse_drag_callbacks.append(self._delete_action)

    def _on_tick_change(self, _):
        if self.cb.isChecked():
            if self.active_layer is not None:
                if self._delete_action not in self.active_layer.mouse_drag_callbacks:
                    if isinstance(self.active_layer, napari.layers.Labels):
                        self.active_layer.mouse_drag_callbacks.append(self._delete_action)
            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.text = '/!\ Removing objects by right-clicking!'
        else:
            for layer in self.viewer.layers:
                if self._delete_action in layer.mouse_drag_callbacks:
                    layer.mouse_drag_callbacks.remove(self._delete_action)
            self.viewer.text_overlay.visible = False

    def _delete_action(self, source_layer, event: Event):
        layer_value = source_layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if layer_value is not None:
            new_labels = source_layer.data
            new_labels[new_labels == layer_value] = 0
            source_layer.data = new_labels
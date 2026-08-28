
from lutuflow.core import initialize_df
from napari.layers import Labels
from PyQt6.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QWidget,
    QPushButton,
)


class ToTracksWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        layout = QGridLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Convert to Tracks
        self.cb_convert_to_tracks = QComboBox()
        layout.addWidget(QLabel("Visualize Tracks", self), 5, 0)
        layout.addWidget(self.cb_convert_to_tracks, 5, 1)
        self.btn_convert_to_tracks = QPushButton("Run", self)
        self.btn_convert_to_tracks.clicked.connect(self._convert_to_tracks)
        layout.addWidget(self.btn_convert_to_tracks, 5, 2)

        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

    def _on_layer_change(self, e):
        self.cb_convert_to_tracks.clear()
        for x in self.viewer.layers:
            if isinstance(x, Labels):
                if len(x.data.shape) == 4:
                    self.cb_convert_to_tracks.addItem(x.name, x.data)
    
    def _convert_to_tracks(self, *args, **kwargs):
        labels_data = self.cb_convert_to_tracks.currentData()
        if labels_data is None:
            return

        df = initialize_df(labels_data, properties=["centroid", "label"])

        # ID, T, Z, Y, X
        tracks_data = df[
            ["label", "frame_forward", "centroid-0", "centroid-1", "centroid-2"]
        ].values

        self.viewer.add_tracks(
            tracks_data,
            head_length=len(labels_data),
            tail_length=len(labels_data),
            name="Tracks",
        )
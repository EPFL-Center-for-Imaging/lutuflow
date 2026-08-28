import napari
import napari.layers
import numpy as np
import pandas as pd
import skimage.measure
from qtpy.QtWidgets import (
    QGridLayout, 
    QWidget, 
    QTableWidget, 
    QTableWidgetItem,
    QCheckBox,
    QLabel,
    QFileDialog,
    QPushButton,
)


class TableWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.selected_labels_layer = None
        self.df = None
        self.current_time = None

        self.setLayout(QGridLayout())

        self.layout().addWidget(QLabel("Follow objects in time", self), 0, 0)
        self.follow_objects_checkbox = QCheckBox()
        self.follow_objects_checkbox.setChecked(False)
        self.layout().addWidget(self.follow_objects_checkbox, 0, 1)

        save_button = QPushButton("Save as CSV")
        save_button.clicked.connect(lambda _: self._save_csv())
        self.layout().addWidget(save_button, 1, 0, 1, 2)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setColumnCount(2)
        self._table.setRowCount(1)
        self._table.setColumnWidth(0, 30)
        self._table.setColumnWidth(1, 120)
        self._table.setHorizontalHeaderItem(0, QTableWidgetItem("label"))
        self._table.setHorizontalHeaderItem(1, QTableWidgetItem("volume"))
        self._table.clicked.connect(self._clicked_table)

        self.layout().addWidget(self._table, 2, 0, 1, 2)

        self.viewer.layers.selection.events.changed.connect(
            self._on_layer_selection_changed
        )
        self.viewer.layers.events.inserted.connect(
            lambda e: self._on_layer_selection_changed(None)
        )
        self._on_layer_selection_changed(None)

    def _on_layer_selection_changed(self, event):
        if event is None:
            selected_layer = self.viewer.layers.selection.active
        else:
            selected_layer = event.source.active

        if isinstance(self.selected_labels_layer, napari.layers.Labels):
            self.selected_labels_layer.events.paint.disconnect(self.update_table_content)
            self.selected_labels_layer.events.data.disconnect(self.update_table_content)
            if self.selected_labels_layer.data.ndim == 4:
                self.viewer.dims.events.current_step.disconnect(self.handle_time_axis_changed)

        if isinstance(selected_layer, napari.layers.Labels):
            selected_layer.events.data.connect(self.update_table_content)
            selected_layer.events.paint.connect(self.update_table_content)
            if selected_layer.data.ndim == 4:
                self.viewer.dims.events.current_step.connect(self.handle_time_axis_changed)

        self.selected_labels_layer = selected_layer

        self.update_table_content()

    @property
    def axes(self):
        if self.viewer.dims.ndisplay == 3:
            return

        # 2D case
        axes = list(self.viewer.dims.displayed)

        # 3D case
        if self.selected_labels_layer.data.ndim == 3:
            axes.insert(
                0,
                list(set([0, 1, 2]) - set(list(self.viewer.dims.displayed)))[
                    0
                ],
            )

        # 4D case (not used yet)
        elif self.selected_labels_layer.data.ndim == 4:
            xxx = set(self.viewer.dims.displayed)
            to_add = list(set([0, 1, 2, 3]) - xxx)
            axes = to_add + axes

        return axes

    def _clicked_table(self):
        if self.selected_labels_layer is None:
            return
        
        self.handle_selected_table_label_changed(self.df["label"].values[self._table.currentRow()])

    def handle_selected_table_label_changed(self, selected_table_label):

        if not selected_table_label in self.df['label'].unique():
            print(f"Label {selected_table_label} is not present.")
            return

        self.selected_labels_layer.selected_label = selected_table_label

        sub_df = self.df[self.df['label'] == selected_table_label]

        x0 = int(sub_df['bbox-0'].values[0])
        x1 = int(sub_df['bbox-3'].values[0])
        y0 = int(sub_df['bbox-1'].values[0])
        y1 = int(sub_df['bbox-4'].values[0])
        z0 = int(sub_df['bbox-2'].values[0])
        z1 = int(sub_df['bbox-5'].values[0])

        label_size = max(x1 - x0, y1 - y0, z1 - z0)

        centers = np.array([(x1 + x0) / 2, (y1 + y0) / 2, (z1 + z0) / 2])

        # Note - there is probably something easier to set up with viewer.camera.calculate_nd_view_direction()
        if self.viewer.dims.ndisplay == 3:
            self.viewer.camera.center = (0.0, centers[1], centers[2])
            self.viewer.camera.angles = (0.0, 0.0, 90.0)
        else:
            current_center = np.array(self.viewer.camera.center)

            if len(self.axes) == 2:
                current_center[1] = centers[1:][self.axes][0]
                current_center[2] = centers[1:][self.axes][1]
            elif len(self.axes) == 3:
                current_center[1] = centers[self.axes[1]]
                current_center[2] = centers[self.axes[2]]
                # In 3D, also adjust the current step
                current_step = np.array(self.viewer.dims.current_step)[
                    self.axes
                ]
                current_step[self.axes[0]] = int(centers[self.axes[0]])
                self.viewer.dims.current_step = tuple(current_step)

            elif len(self.axes) == 4:
                # TODO - This is very experimental (probably not working when layers are transposed)
                current_center[1] = centers[self.axes[2]-1]
                current_center[2] = centers[self.axes[3]-1]
                current_step = np.array(self.viewer.dims.current_step)[
                    self.axes
                ]
                current_step[self.axes[1]] = int(centers[self.axes[1]-1])
                self.viewer.dims.current_step = tuple(current_step)

            self.viewer.camera.center = tuple(current_center)

        self.viewer.camera.zoom = max(3 - label_size * 0.005, 1.0)

    def updated_content_2D_or_3D(self, labels):
        """Compute volumes and update the table UI in the 2D and 3D cases."""
        properties = skimage.measure.regionprops_table(
            labels, properties=["label", "area", "bbox"]
        )
        self.df = pd.DataFrame.from_dict(properties)
        self.df.rename(columns={"area": "volume"}, inplace=True)
        self.df.sort_values(by="volume", ascending=False, inplace=True)

        # Regenerate the table UI
        self._table.clear()
        self._table.setRowCount(len(self.df))
        self._table.setHorizontalHeaderItem(0, QTableWidgetItem("label"))
        self._table.setHorizontalHeaderItem(1, QTableWidgetItem("volume"))

        k = 0
        for _, (lab, vol) in self.df[["label", "volume"]].iterrows():
            self._table.setItem(k, 0, QTableWidgetItem(str(lab)))
            self._table.setItem(k, 1, QTableWidgetItem(str(vol)))
            k += 1

    def handle_time_axis_changed(self, event):
        current_time = event.value[0]
        if (current_time != self.current_time) | (self.current_time is None):
            self.current_time = current_time
            current_selected_label = self.selected_labels_layer.selected_label
            self.update_table_content()
            if self.follow_objects_checkbox.isChecked():
                self.handle_selected_table_label_changed(current_selected_label)

    def update_table_content(self):
        if not isinstance(self.selected_labels_layer, napari.layers.Labels):
            self._table.clear()
            self._table.setRowCount(1)
            self._table.setColumnWidth(0, 30)
            self._table.setColumnWidth(1, 120)
            self._table.setHorizontalHeaderItem(0, QTableWidgetItem("label"))
            self._table.setHorizontalHeaderItem(1, QTableWidgetItem("volume"))
            return

        labels = self.selected_labels_layer.data

        if len(labels.shape) == 2:
            labels = labels[None]  # Add an extra dimension in the 2D case

        elif len(labels.shape) == 4:
            labels = labels[self.viewer.dims.current_step[0]]

        if labels.sum() == 0:
            return

        self.updated_content_2D_or_3D(labels)

    def _save_csv(self):
        if self.df is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as CSV", ".", "*.csv"
        )

        pd.DataFrame(self.df[['label', 'volume']]).to_csv(filename)

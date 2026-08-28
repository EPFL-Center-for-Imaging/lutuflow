import napari
from lutuflow.napari_tools import OMEROWidget
from lutuflow import __version__

if __name__ == "__main__":
    viewer = napari.Viewer(title=f"LuTuFlow ({__version__})")
    viewer.window.add_dock_widget(OMEROWidget(viewer), name="LuTuFlow")
    napari.run()

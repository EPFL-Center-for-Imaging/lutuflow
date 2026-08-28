![screenshot](./assets/screenshot.png)
# 🐭 LuTuFlow

> A toolbox to segment and track lung tumor nodules in longitudinal series of mice CT scans.

## Hightlights

- **Detect the lungs automatically** using a pretrained [YoloV8](https://docs.ultralytics.com/) model and crop the scans around them.
- **Detect tumor nodules** using a pretrained [nnUNet](https://github.com/MIC-DKFZ/nnUNet) 3D segmentation model.
- **Track individual tumors** across several CT scans of the same mouse.

We provide a unified user interface in Napari to detect, track, visualize, annotate, and measure the size evolution of lung tumor nodules in mice CT scans. The datasets and experiment metadata are automatically downloaded and parsed from OMERO.

This project is part of a collaboration between the [EPFL Center for Imaging](https://imaging.epfl.ch/) and the [De Palma Lab](https://www.epfl.ch/labs/depalma-lab/).

## Installation

**As a standalone app**

Download and run the latest installer from the [Releases](https://github.com/EPFL-Center-for-Imaging/lutuflow/releases) page. This is the simplest option, but it only allows usage in Napari (not as a CLI).

**In Python**

We recommend performing the installation in a clean Python environment. First, install the `zeroc-ice` package via the pre-built wheels from Glencoe Software. Choose the wheel corresponding to your python version (3.10, 3.11, 3.12) and platform (Windows, MacOS, Linux).

- Windows: https://github.com/glencoesoftware/zeroc-ice-py-win-x86_64/releases/
- MacOS: https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases/
- Linux: https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases

Then, install our package from PyPi:

```sh
pip install lutuflow
```

or from the repository:

```sh
pip install git+https://github.com/EPFL-Center-for-Imaging/lutuflow.git
```

or clone the repository and install with:

```sh
git clone git+https://github.com/EPFL-Center-for-Imaging/lutuflow.git
cd lutuflow
pip install -e .
```

## Usage

### In Napari

From the command-line, start Napari:

```
napari
```

Then, access the plugin's functionality in Napari via the menu `Plugins > LuTuFlow`. Several tools are available:

- `OMERO interface`: Interface to use LuTuFlow with OMERO. Refer to the [documentation](https://github.com/EPFL-Center-for-Imaging/lutuflow/wiki) for more details.
- `LuTuFlow`: Run individual workflow steps:
  - `1. Crop original scans`: Automatically crop 3D CT scans around the lungs ROI.
  - `2. Segment tumors`: Segments tumor nodules in a CT scan image of the lungs ROI.
  - `3. Combine CT scans`: Create a 4D time series (TZYX) from CT images.
  - `3. Combine tumor masks`: Create a 4D time series (TZYX) from tumor masks.
  - `4. Track tumors`: Tracks tumors in a 4D tumor masks series.
  - `5. Save tracking table`: Save tracking results as a CSV file.
- `Color picker`: Display a segmentation mask in a chosen color.
- `Manual cropping`: Crop CT scans by setting limits in the X, Y and Z directions.
- `Convert to tracks`: Displays tracks from a segmentation mask layer.
- `Data table`: Display tumor labels and volumes in a table.
- `Remove objects`: Remove individual tumors by clicking on them in the viewer.

#### Sample images

We provide two sample images under `File > Open Sample > LuTuFlow` to test the package's functionality:

- **Mouse lung CT scan**: An example CT scan, in its original size (shape: (512, 512, 512)). This image can be used to test the steps `1. Crop original scans` followed by `2. Segment tumors`.
- **Lung ROI series (TZYX)**: An example of 5 concatenated scans, after ROI extraction (shape: (5, 262, 349, 330)). This image can be used to test the steps `2. Segment tumors` followed by `4. Track tumors` and `5. Save tracking table`.

### As a CLI

Several functions from `lutuflow` can be used from the command-line interface (CLI).

**OMERO**

Run LuTuFlow in batch and interact with OMERO projects.

In interactive mode:

```
lutuflow omero interactive
```

To run the tumor detection workflow on an OMERO project:

```
lutuflow omero run <project_id> --tumor-model oct24
```

For more details, see `lutuflow omero --help`.

**Crop**

Run a [YoloV8](https://docs.ultralytics.com/) model to segment the lungs cavity and crop the image around the lungs.

```sh
lutuflow crop <image_file> <out_dir>
```

For more details, see `lutuflow crop --help`.

**Predict**

Run a [nnUNet](https://github.com/MIC-DKFZ/nnUNet) model to segment tumor nodules.

```sh
lutuflow predict <image_file> <out_dir>
```

For more details, see `lutuflow predict --help`.

**Combine**

Combine several 3D images (ZYX) into a single 4D image (TZYX).  

```sh
lutuflow combine <image_1> <image_2> <image_3> <out_dir>
```

For more details, see `lutuflow combine --help`.

**Track**

Track tumor nodules across a 4D mask (TZYX) time series using [laptrack](https://github.com/yfukai/laptrack).

```sh
lutuflow track <tumors_file> <image_file> <lungs_file> <out_dir>
```

For more details, see `lutuflow track --help`.

**Serve**

```sh
lutuflow serve --port 8000
```

Create an [Imaging Server Kit](https://github.com/Imaging-Server-Kit/imaging-server-kit) server to use LuTuFlow functions remotely.

For more details, see `lutuflow serve --help`.

## License

This project is licensed under the [AGPL-3](LICENSE) license.

This project depends on the [ultralytics](https://github.com/ultralytics/ultralytics) package which is licensed under AGPL-3.

This project uses the [PyApp](https://github.com/ofek/pyapp) software for creating a runtime installer.
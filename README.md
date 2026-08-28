![screenshot](./assets/screenshot.png)
# 🐭 LuTuFlow

LuTuFlow is a Python toolbox for segmenting and tracking lung tumor nodules in longitudinal series of mice CT scans. It is developed as a collaboration between the [EPFL Center for Imaging](https://imaging.epfl.ch/) and the [De Palma Lab](https://www.epfl.ch/labs/depalma-lab/).

## Hightlights

- **Detect the lungs automatically** using a pretrained [YoloV8](https://docs.ultralytics.com/) model and crop the scans around them.
- **Detect tumor nodules** using a pretrained [nnUNet](https://github.com/MIC-DKFZ/nnUNet) 3D segmentation model.
- **Track individual tumors** across several CT scans of the same mouse.

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

LuTuFlow can be used in Napari or from the command-line. See the [documentation](https://github.com/EPFL-Center-for-Imaging/lutuflow/wiki) for usage instructions.

## License

This project is licensed under the [AGPL-3](LICENSE) license.

This project depends on the [ultralytics](https://github.com/ultralytics/ultralytics) package which is licensed under AGPL-3.

This project uses the [PyApp](https://github.com/ofek/pyapp) software for creating a runtime installer.
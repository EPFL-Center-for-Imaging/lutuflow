import os
from pathlib import Path
import glob
import shutil
import subprocess
import pooch
import torch
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import skimage.morphology
import skimage.io
from lutuflow.core.configuration import MIN_SIZE_PX, NNUNET_MODELS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TumorPredictor:
    """
    Predictor for tumor segmentation using pre-trained nnUNet models.

    Parameters
    ----------
    model : str
        Identifier of the nnUNet model to use. Must be a key in configuration.NNUNET_MODELS.
    
    Methods
    ----------
    predict(): Segment the given 3D image using the loaded nnUNet model.
    """
    def __init__(self, model: str):
        self.model = model
        model_path = str(Path.home() / ".lutuflow" / model)

        model_url, model_known_hash = NNUNET_MODELS.get(model)

        pooch.retrieve(
            url = model_url,
            # known_hash= model_known_hash,
            known_hash=None,
            path=model_path,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=model_path)
        )

        os.environ["nnUNet_results"] = model_path

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the given 3D image using the loaded nnUNet model.

        Parameters
        ----------
        image : np.ndarray
            3D image array for tumor segmentation.

        Returns
        -------
        np.ndarray
            Labeled segmentation mask with connected tumor components.
        """
        model_path = os.getenv("nnUNet_results")

        self.input_folder = os.path.join(model_path, "tmp", "nnunet_input")
        self.output_folder = os.path.join(model_path, "tmp", "nnunet_output")

        if not os.path.exists(self.input_folder): 
            os.makedirs(self.input_folder)
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        if self.model in ["nov23", "oct24"]:
            nib.save(nib.Nifti1Image(image, None), os.path.join(self.input_folder, "img_0000.nii.gz"))
        else:
            skimage.io.imsave(os.path.join(self.input_folder, "img_0000.tif"), image)

        try:
            self._run_nnunet(device=DEVICE)
        except subprocess.CalledProcessError as e:
            if DEVICE == "cuda":
                print("Could not run nnUNet with CUDA even if it is available. Falling back to CPU instead...")
                self._run_nnunet(device="cpu")
            else:
                raise
        
        if self.model in ["nov23", "oct24"]:  # Legacy models, trained on nii format
            output_preds_file = list(glob.glob(os.path.join(self.output_folder, "*.gz")))
            if len(output_preds_file) > 0:
                segmentation = nib.load(output_preds_file[0]).get_fdata().astype(np.uint16)
            else:
                raise RuntimeError(f"Running nnUNet (model {self.model}) on this image was unsuccessful.")
        else:  # New models based on tiff format
            output_preds_file = list(glob.glob(os.path.join(self.output_folder, "*.tif")))
            if len(output_preds_file) > 0:
                segmentation = skimage.io.imread(output_preds_file[0]).astype(np.uint16)
            else:
                raise RuntimeError(f"Running nnUNet (model {self.model}) on this image was unsuccessful.")

        shutil.rmtree(str(self.input_folder))
        shutil.rmtree(str(self.output_folder))

        # Label connected components
        segmentation, _ = ndi.label(segmentation)

        # Remove small objects
        segmentation = skimage.morphology.remove_small_objects(segmentation, min_size=MIN_SIZE_PX)

        # Fill holes in each Z slice
        for z in range(segmentation.shape[0]):
            slice_segmentation = segmentation[z]
            filled_slice = ndi.binary_fill_holes(slice_segmentation)
            segmentation[z] = filled_slice
        
        # Label connected components
        segmentation, _ = ndi.label(segmentation)

        segmentation = segmentation.astype(np.uint16)

        return segmentation

    def _run_nnunet(self, device: str = "cpu"):
        return subprocess.run(  # Note: I think this may still hang if cuda runs OOM...
            [
                "nnUNetv2_predict",
                "-i", self.input_folder,
                "-o", self.output_folder,
                "-d", "001",
                "-f", "0",
                "-c", "3d_fullres",
                "-device", device,
                "--disable_tta",
            ],
            check=True,
        )
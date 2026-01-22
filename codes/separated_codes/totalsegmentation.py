import traceback
from totalsegmentator.python_api import totalsegmentator
import os
import tempfile

def run_TS(nifti):
    temp_root = os.path.join(tempfile.gettempdir(), "Pyramid_RAS")
    os.makedirs(temp_root, exist_ok=True)

    out_dir = os.path.join(temp_root, "RAS_output")
    os.makedirs(out_dir, exist_ok=True)

    try:
        totalsegmentator(
            input=nifti,
            output=out_dir,           
            task="total",
            roi_subset=["pancreas"]
        )

        mask_path = os.path.join(out_dir, "pancreas.nii.gz")

        if not os.path.exists(mask_path):
            return None

        return mask_path

    except Exception as e:
        traceback.print_exc()
        return None

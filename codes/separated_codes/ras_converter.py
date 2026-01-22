import os
import tempfile
import SimpleITK as sitk


def dicom_to_nifti_ras(dicom_folder):
    
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise ValueError(f"No DICOM series found: {dicom_folder}")
    
    best_id = max(series_ids, key=lambda s: len(reader.GetGDCMSeriesFileNames(dicom_folder, s)))
    files = reader.GetGDCMSeriesFileNames(dicom_folder, best_id)

    reader.SetFileNames(files)
    img = reader.Execute()

    ras_img = sitk.DICOMOrient(img, "RAS")

    temp_root = os.path.join(tempfile.gettempdir(), "Pyramid_RAS")
    os.makedirs(temp_root, exist_ok=True)

    base_name = os.path.basename(dicom_folder.rstrip("/\\"))
    out_path = os.path.join(temp_root, f"{base_name}_RAS.nii.gz")

    sitk.WriteImage(ras_img, out_path)

    return out_path

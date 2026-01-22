import os
import pydicom

def collect_patient_information(dicom_folder):
    if not os.path.isdir(dicom_folder):
        raise ValueError(f"경로가 폴더가 아닙니다: {dicom_folder}")

    dicom_files = []
    for root, dirs, files in os.walk(dicom_folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    if len(dicom_files) == 0:
        raise FileNotFoundError("DICOM 파일을 찾을 수 없습니다.")

    # 3. 첫 번째 DICOM 읽기
    dcm_path = dicom_files[0]
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    # 4. 환자 이름 가져오기
    name = getattr(ds, "PatientName", "Unknown")

    return str(name)

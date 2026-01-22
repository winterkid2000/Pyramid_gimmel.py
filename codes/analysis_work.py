class AnalysisWorker(QObject):
    finished = Signal(str, str, object)  # nifti_path, mask_path, top_features_df
    log = Signal(str)
    error = Signal(str)

    def __init__(self, dicom_path, threshold):
        super().__init__()
        self.dicom_path = dicom_path
        self.threshold = threshold

   
    def run(self):
        try:
            self.log.emit("[1] 환자 성함 확인 중...")
            name = collect_patient_information(self.dicom_path)

            self.log.emit("[2] DICOM → NIfTI 변환 중...")
            nifti_path = dicom_to_nifti_ras(self.dicom_path)

            self.log.emit("[3] 췌장 segmentation 중...")
            mask_path = run_TS(nifti_path)
        
            self.log.emit("[4] Radiomics 추출 중...")
            yaml_path = r'c:\Users\RaPhyA\Desktop\Nous\assets\parameters.yaml'
            radiomics = extract_radiomics(nifti_path, mask_path, yaml_path)

            self.log.emit("[5] AI 예측 중...")
            model_path = r'c:\Users\RaPhyA\Desktop\Nous\assets\final_model.pt'
            scaler_path = r'c:\Users\RaPhyA\Desktop\Nous\assets\scaler.pkl'

            result_df, top_features_df = predict_with_model(
                radiomics,
                name,
                model_path,
                scaler_path,
                threshold=self.threshold,
                log_callback=self.log.emit
            )

            self.finished.emit(nifti_path, mask_path, top_features_df)

        except Exception as e:
            self.error.emit(str(e))
            self.error.emit(traceback.format_exc())

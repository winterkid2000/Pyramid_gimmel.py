class RepredictWorker(QObject):
    finished = Signal(object)  # top_features_df
    log = Signal(str)
    error = Signal(str)

    def __init__(self, nifti_path, mask_vol, patient_name, threshold):
        super().__init__()
        self.nifti_path = nifti_path
        self.mask_vol = mask_vol
        self.patient_name = patient_name
        self.threshold = threshold

    def run(self):
        # 임시 파일 경로 변수 초기화 (finally 블록에서 삭제하기 위해)
        mask_path = None
        
        try:
            # 편집된 마스크를 임시 파일로 저장
            self.log.emit("💾 편집된 마스크 저장 중...")
            import tempfile
            # delete=False로 해야 윈도우에서 파일 잠금 문제를 피할 수 있음
            temp_mask = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
            mask_path = temp_mask.name
            temp_mask.close()  # 중요: 파일을 닫아야 다른 프로세스(nibabel 등)가 접근 가능
            
            # NIfTI 파일의 affine과 header를 유지하면서 저장
            nifti_img = nib.load(self.nifti_path)
            # 마스크 데이터 타입 확인 (uint8 권장)
            mask_img = nib.Nifti1Image(self.mask_vol.astype(np.uint8), nifti_img.affine, nifti_img.header)
            nib.save(mask_img, mask_path)

            self.log.emit("[4] Radiomics 재추출 중...")
            yaml_path = r'c:\Users\RaPhyA\Desktop\Nous\assets\parameters.yaml'
            radiomics = extract_radiomics(self.nifti_path, mask_path, yaml_path)

            self.log.emit("[5] AI 재예측 중...")
            model_path = r'c:\Users\RaPhyA\Desktop\Nous\assets\final_model.pt'
            scaler_path = r'c:\Users\RaPhyA\Desktop\Nous\assets\scaler.pkl'

            result_df, top_features_df = predict_with_model(
                radiomics,
                self.patient_name,
                model_path,
                scaler_path,
                threshold=self.threshold,
                log_callback=self.log.emit
            )
            
            # 성공 로그는 여기서 찍어도 되지만, finished 연결된 곳에서 찍어도 됨

        except Exception as e:
            self.error.emit(f"❌ 에러 발생: {str(e)}")
            self.error.emit(traceback.format_exc())
            top_features_df = None
            
        finally:
            # 1. 임시 파일 삭제 (에러가 나도 파일은 지워야 함)
            if mask_path and os.path.exists(mask_path):
                try:
                    os.unlink(mask_path)
                except Exception:
                    pass
            
            # 2. ★핵심 수정★: 에러가 나든 안 나든 반드시 종료 시그널 전송
            self.finished.emit(top_features_df)

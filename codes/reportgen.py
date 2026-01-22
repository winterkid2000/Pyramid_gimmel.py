class ReportGenWorker(QObject):
    finished = Signal(str)      
    error = Signal(str)         
    log = Signal(str)           

    def __init__(self, generator_instance, shap_df, feature_dict=None):
        super().__init__()
        self.generator = generator_instance
        self.shap_df = shap_df
        self.feature_dict = feature_dict

    def run(self):
        try:
            
            if self.generator is None:
                self.log.emit("🔄 Loading BioMistral model (First run takes time)...")
                
                self.generator = BioMistralReportGenerator()
                self.log.emit("✅ Model loaded!")

            # 2. 리포트 생성
            self.log.emit("✍️ Generating report...")
            report = self.generator.generate_report(
                shap_df=self.shap_df,
                feature_dictionary=self.feature_dict,
                top_n=min(10, len(self.shap_df)),
                max_tokens=2048,
                temperature=0.7
            )
            
            self.finished.emit(report)

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

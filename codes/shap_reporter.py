class ShapGraphDialog(QDialog):
    def __init__(self, top_features_df, patient_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"SHAP 분석 결과 - {patient_name}")
        self.setGeometry(300, 200, 1200, 700)

        self.report_thread = None
        self.report_worker = None
        self.biomistral_generator = None  # 생성기 인스턴스 저장소

        # Main horizontal layout (그래프 | 리포트)
        main_layout = QHBoxLayout(self)
        
        # ===== Left: SHAP Graph =====
        left_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)
        self.plot_shap(top_features_df, patient_name)
        main_layout.addLayout(left_layout, 6)
        
        # ===== Right: BioMistral Report =====
        right_layout = QVBoxLayout()
        report_title = QLabel("🤖 AI Generated Radiological Report")
        report_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        right_layout.addWidget(report_title)
        
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlainText("Click 'Generate Report' to analyze SHAP values with BioMistral.")
        self.report_text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 10pt; padding: 10px;")
        right_layout.addWidget(self.report_text)
        
        self.generate_btn = QPushButton("🔄 Generate Report")
        # 람다 대신 메서드 직접 연결
        self.generate_btn.clicked.connect(self.start_report_generation) 
        self.generate_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        right_layout.addWidget(self.generate_btn)
        
        main_layout.addLayout(right_layout, 4)
        
        # ===== Bottom: Close button =====
        bottom_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close_dialog) # 닫을 때 스레드 정리를 위해 메서드 변경
        close_btn.setStyleSheet("padding: 8px;")
        bottom_layout.addStretch()
        bottom_layout.addWidget(close_btn)
        
        full_layout = QVBoxLayout()
        full_layout.addLayout(main_layout)
        full_layout.addLayout(bottom_layout)
        self.setLayout(full_layout)
        
        self.top_features_df = top_features_df

    def start_report_generation(self):
        """스레드를 시작하여 리포트를 생성합니다."""
        # UI 상태 변경
        self.generate_btn.setEnabled(False)
        self.report_text.setPlainText("⏳ Initializing process...\n")
        
        # 기존 스레드 정리
        if self.report_thread is not None:
            if self.report_thread.isRunning():
                self.report_thread.quit()
                self.report_thread.wait()
            self.report_thread.deleteLater()
            
        # SHAP DataFrame 준비
        shap_df = pd.DataFrame({
            'feature': self.top_features_df['Feature'].values,
            'shap_value': self.top_features_df['SHAP_Value'].values
        })

        # Worker 및 Thread 설정
        self.report_thread = QThread()
        # generator 인스턴스를 전달 (없으면 Worker 내부에서 생성됨)
        self.report_worker = ReportGenWorker(self.biomistral_generator, shap_df)
        self.report_worker.moveToThread(self.report_thread)
        
        # 시그널 연결
        self.report_thread.started.connect(self.report_worker.run)
        self.report_worker.log.connect(self.update_log)
        self.report_worker.finished.connect(self.on_report_success)
        self.report_worker.error.connect(self.on_report_error)
        
        # 종료 처리
        self.report_worker.finished.connect(self.report_thread.quit)
        self.report_worker.finished.connect(self.report_worker.deleteLater)
        self.report_thread.finished.connect(self.report_thread.deleteLater)
        
        # 스레드 시작
        self.report_thread.start()

    def update_log(self, message):
        """진행 상황을 텍스트 박스에 표시"""
        self.report_text.appendPlainText(message)

    def on_report_success(self, report_content):
        """생성 성공 시 호출"""
        self.generate_btn.setEnabled(True)
        
        # 생성된 generator 인스턴스를 저장해둠 (다음 번 클릭 시 로딩 시간 단축)
        if self.biomistral_generator is None:
             self.biomistral_generator = self.report_worker.generator

        self.report_text.setPlainText("=" * 60 + "\n")
        self.report_text.appendPlainText("BIOMISTRAL RADIOLOGICAL REPORT\n")
        self.report_text.appendPlainText("=" * 60 + "\n\n")
        self.report_text.appendPlainText(report_content)
        self.report_text.appendPlainText("\n\n" + "=" * 60)
        self.report_text.appendPlainText("\n⚠️ This report is AI-generated and should be reviewed by a medical professional.")

    def on_report_error(self, error_msg):
        """에러 발생 시 호출"""
        self.generate_btn.setEnabled(True)
        self.report_text.setPlainText(f"❌ Error generating report:\n\n{error_msg}\n")
        self.report_text.appendPlainText("\nPossible solutions:\n")
        self.report_text.appendPlainText("1. Check VRAM/RAM availability.\n")
        self.report_text.appendPlainText("2. Check internet for model download.\n")

    def close_dialog(self):
        """다이얼로그 닫을 때 스레드 안전하게 종료"""
        if self.report_thread is not None and self.report_thread.isRunning():
            self.report_thread.quit()
            self.report_thread.wait(1000)
        self.accept()
        
    # plot_shap 메서드는 기존 그대로 유지
    def plot_shap(self, df, patient_name):
        # ... (기존 코드와 동일) ...
        self.ax.clear()
        features = df['Feature'].values
        shap_values = df['SHAP_Value'].values
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_values]
        y_pos = np.arange(len(features))
        self.ax.barh(y_pos, shap_values, color=colors, alpha=0.7, edgecolor='black')
        feature_labels = [f[:30] + '...' if len(f) > 30 else f for f in features]
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(feature_labels)
        self.ax.set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
        self.ax.set_title('Dr. Pyramid\'s Report', fontsize=14, fontweight='bold', pad=20)
        self.ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        self.ax.grid(axis='x', alpha=0.3, linestyle='--')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.7, label='Positive'),
            Patch(facecolor='#3498db', alpha=0.7, label='Negative')
        ]
        self.ax.legend(handles=legend_elements, loc='lower right')
        self.figure.tight_layout()
        self.canvas.draw()

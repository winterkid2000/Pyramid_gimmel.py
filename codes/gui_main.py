class RadiomicsAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pyramid(Py-RadioMics-Detector)")
        self.setGeometry(200, 100, 1500, 900)

    
        self.thread = None
        self.worker = None
        self.repredict_thread = None
        self.repredict_worker = None

        self.nifti_vol = None
        self.mask_vol = None
        self.current_view = "axial"
        self.zoom_factor = 1.0
        
        # For repredict
        self.nifti_path = None
        self.patient_name = None

        # Graphics scene setup
        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Edit item for mask editing
        self.edit_item = EditContourItem(diameter=10, colour=Qt.red, mode="erase")
        self.scene.addItem(self.edit_item)

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # ================= Left (Viewer) =================
        left_layout = QVBoxLayout()

        self.viewer = ImageView(self.scene, self.edit_item)
        self.viewer.setStyleSheet("background:black;")
        self.viewer.wheel_slice.connect(self.on_wheel_slice)
        self.viewer.wheel_zoom.connect(self.on_zoom)

        left_layout.addWidget(self.viewer)

        # View change buttons
        btn_layout = QHBoxLayout()
        for txt, v in [("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal")]:
            b = QPushButton(txt)
            b.clicked.connect(lambda _, vv=v: self.change_view(vv))
            btn_layout.addWidget(b)
        left_layout.addLayout(btn_layout)

        # Edit controls
        edit_box = QGroupBox("✏️ 마스크 편집")
        edit_layout = QVBoxLayout()

        # Edit mode toggle
        edit_mode_layout = QHBoxLayout()
        self.edit_mode_btn = QPushButton("편집 모드")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.toggled.connect(self.toggle_edit_mode)
        edit_mode_layout.addWidget(self.edit_mode_btn)

        # Erase/Add radio buttons
        self.erase_radio = QRadioButton("지우기")
        self.erase_radio.setChecked(True)
        self.erase_radio.toggled.connect(lambda: self.edit_item.set_mode("erase"))
        self.add_radio = QRadioButton("추가")
        self.add_radio.toggled.connect(lambda: self.edit_item.set_mode("add"))
        edit_mode_layout.addWidget(self.erase_radio)
        edit_mode_layout.addWidget(self.add_radio)
        edit_layout.addLayout(edit_mode_layout)

        # Brush size
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("브러시 크기:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(5, 50)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(self.edit_item.set_brush_size)
        self.brush_size_label = QLabel("10")
        self.brush_slider.valueChanged.connect(lambda v: self.brush_size_label.setText(str(v)))
        brush_layout.addWidget(self.brush_slider)
        brush_layout.addWidget(self.brush_size_label)
        edit_layout.addLayout(brush_layout)

        # Apply/Clear buttons
        apply_layout = QHBoxLayout()
        self.apply_edit_btn = QPushButton("편집 적용")
        self.apply_edit_btn.clicked.connect(self.apply_edits)
        self.clear_edit_btn = QPushButton("편집 취소")
        self.clear_edit_btn.clicked.connect(self.clear_edits)
        apply_layout.addWidget(self.apply_edit_btn)
        apply_layout.addWidget(self.clear_edit_btn)
        edit_layout.addLayout(apply_layout)
        
        # Repredict button
        self.repredict_btn = QPushButton("🔄 재예측")
        self.repredict_btn.clicked.connect(self.repredict)
        self.repredict_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        edit_layout.addWidget(self.repredict_btn)

        edit_box.setLayout(edit_layout)
        left_layout.addWidget(edit_box)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.update_slice_view)
        left_layout.addWidget(self.slice_slider)

        # ================= Right (Control) =================
        right_layout = QVBoxLayout()

        dicom_box = QGroupBox("📂 DICOM 폴더 선택")
        d_l = QVBoxLayout()
        self.dicom_label = QLabel("선택된 폴더 없음")
        b = QPushButton("폴더 선택")
        b.clicked.connect(self.select_folder)
        d_l.addWidget(self.dicom_label)
        d_l.addWidget(b)
        dicom_box.setLayout(d_l)

        mode_box = QGroupBox("⚙️ 분석 모드")
        m_l = QHBoxLayout()
        self.mode_slider = QSlider(Qt.Horizontal)
        self.mode_slider.setRange(0, 1)
        self.mode_label = QLabel("표준 모드")
        self.mode_slider.valueChanged.connect(
            lambda v: self.mode_label.setText(["표준 모드", "고감도 모드"][v])
        )
        m_l.addWidget(self.mode_slider)
        m_l.addWidget(self.mode_label)
        mode_box.setLayout(m_l)

        self.run_btn = QPushButton("분석 시작 ▶")
        self.run_btn.clicked.connect(self.run_analysis)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)

        right_layout.addWidget(dicom_box)
        right_layout.addWidget(mode_box)
        right_layout.addWidget(self.run_btn)
        right_layout.addWidget(self.log_box, 1)

        layout.addLayout(left_layout, 7)
        layout.addLayout(right_layout, 3)

    # ----------------------------------------------------------
    def log(self, msg):
        self.log_box.appendPlainText(str(msg))
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    # ----------------------------------------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.dicom_path = folder
            self.dicom_label.setText(folder)

    def cleanup_thread(self, thread_attr_name, worker_attr_name):
        """이전 스레드와 워커를 안전하게 정리 (방어적 코딩 적용)"""
        
        # 1. 워커(Worker) 정리
        if hasattr(self, worker_attr_name):
            worker = getattr(self, worker_attr_name)
            if worker is not None:
                try:
                    # C++ 객체가 살아있는지 확인 (Deleted 객체 접근 방지)
                    worker.disconnect() 
                    worker.deleteLater()
                except RuntimeError:
                    pass # 이미 삭제된 객체면 무시
                except Exception as e:
                    print(f"Worker cleanup error: {e}")
            
            # Python 변수 초기화
            setattr(self, worker_attr_name, None)

        # 2. 스레드(Thread) 정리
        if hasattr(self, thread_attr_name):
            thread = getattr(self, thread_attr_name)
            if thread is not None:
                try:
                    # 스레드가 실행 중인지 안전하게 확인
                    if thread.isRunning():
                        thread.quit()
                        # wait를 너무 길게 잡으면 GUI가 멈춤 -> 100ms로 단축
                        if not thread.wait(100): 
                            thread.terminate() # 안 꺼지면 강제 종료
                            thread.wait(10)
                    
                    thread.deleteLater()
                except RuntimeError:
                    pass # 이미 삭제된 객체면 무시
                except Exception as e:
                    print(f"Thread cleanup error: {e}")
            
            # Python 변수 초기화
            setattr(self, thread_attr_name, None)
        
        # 메모리 정리는 너무 자주하면 렉 걸리므로, 필요할 때만 수행하거나 제거
        # import gc
        # gc.collect()
    # ----------------------------------------------------------
    def run_analysis(self):
        if not hasattr(self, "dicom_path"):
            self.log("❌ DICOM 폴더를 선택하세요.")
            return
        
        # [수정됨] 새 분석 시작 전, 기존의 분석 스레드 AND 재예측 스레드 모두 청소
        self.log("🧹 새 분석을 위해 메모리 정리 중...")
        self.cleanup_thread('thread', 'worker')             # 메인 분석 스레드 정리
        self.cleanup_thread('repredict_thread', 'repredict_worker') # 재예측 스레드도 정리
        
        try:
            self.patient_name = collect_patient_information(self.dicom_path)
        except:
            self.patient_name = "Unknown"

        mode = self.mode_slider.value()
        threshold = 0.5 if mode == 0 else 0.3748581

        self.log(f"=== 분석 시작: {self.patient_name} ===")

        self.thread = QThread()
        self.worker = AnalysisWorker(self.dicom_path, threshold)
        self.worker.moveToThread(self.thread)

        self.worker.log.connect(self.log)
        self.worker.error.connect(lambda e: self.log("오류: " + e))
        self.worker.finished.connect(self.on_finished)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        
        # 스레드 종료 시 자동 삭제 예약
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater)

        self.thread.start()

    # ----------------------------------------------------------
    def on_finished(self, nifti, mask, top_features_df):
        self.log("🎉 분석 완료! 영상 로드 중...")
        self.nifti_path = nifti
        self.load_volumes(nifti, mask)
        self.update_slice_view()
        
        # SHAP 그래프 팝업 표시
        if top_features_df is not None and len(top_features_df) > 0:
            dialog = ShapGraphDialog(top_features_df, self.patient_name or "환자", self)
            dialog.exec()

    # ----------------------------------------------------------
    def load_volumes(self, nifti_path, mask_path):
        self.nifti_vol = nib.load(nifti_path).get_fdata()
        self.mask_vol = nib.load(mask_path).get_fdata()

        idx = np.where(self.mask_vol.sum(axis=(0, 1)) > 0)[0]
        smin = max(0, idx.min() - 1)
        smax = min(self.mask_vol.shape[2] - 1, idx.max() + 1)

        self.slice_slider.setMinimum(smin)
        self.slice_slider.setMaximum(smax)
        self.slice_slider.setValue((smin + smax) // 2)

    # ----------------------------------------------------------
    def on_wheel_slice(self, d):
        v = self.slice_slider.value() + d
        v = max(self.slice_slider.minimum(), min(self.slice_slider.maximum(), v))
        self.slice_slider.setValue(v)

    def on_zoom(self, s):
        self.zoom_factor *= s
        self.update_slice_view()

    # ----------------------------------------------------------
    def change_view(self, v):
        self.current_view = v
        self.update_slice_view()

    # ----------------------------------------------------------
    def toggle_edit_mode(self, enabled):
        self.viewer.set_edit_mode(enabled)
        if enabled:
            self.log("✏️ 편집 모드 활성화")
        else:
            self.log("👁️ 보기 모드 활성화")

    def clear_edits(self):
        self.edit_item.clear()
        self.log("🗑️ 편집 내용 취소")
    
    def repredict(self):
        if self.mask_vol is None or self.nifti_path is None:
            self.log("❌ 재예측할 데이터가 없습니다.")
            return

        # [수정됨] 시작 전 무조건 이전 재예측 스레드 청소
        self.log("🧹 메모리 정리 중...")
        self.cleanup_thread('repredict_thread', 'repredict_worker')

        mode = self.mode_slider.value()
        threshold = 0.5 if mode == 0 else 0.3748581
        
        self.log(f"=== 재예측 시작 (환자: {self.patient_name}) ===")
        
        self.repredict_thread = QThread()
        self.repredict_worker = RepredictWorker(
            self.nifti_path,
            self.mask_vol.copy(),
            self.patient_name or "Unknown",
            threshold
        )
        self.repredict_worker.moveToThread(self.repredict_thread)
        
        self.repredict_worker.log.connect(self.log)
        # 에러 발생 시 로그 찍고 스레드 종료
        self.repredict_worker.error.connect(lambda e: self.log(f"오류: {e}"))
        self.repredict_worker.error.connect(self.repredict_thread.quit) 
        
        self.repredict_worker.finished.connect(self.repredict_thread.quit)
        self.repredict_worker.finished.connect(self.on_repredict_finished)
        
        # 스레드 시작 시 워커 실행
        self.repredict_thread.started.connect(self.repredict_worker.run)
        
        # 스레드가 끝나면 객체 삭제 예약 (중요)
        self.repredict_thread.finished.connect(self.repredict_thread.deleteLater)
        self.repredict_worker.finished.connect(self.repredict_worker.deleteLater)
        
        self.repredict_thread.start()
    
    def on_repredict_finished(self, top_features_df):
        """재예측 완료 후 SHAP 그래프 팝업 표시"""
        self.log("✅ 재예측 완료!")
        
        # SHAP 그래프 팝업 표시
        if top_features_df is not None and len(top_features_df) > 0:
            dialog = ShapGraphDialog(top_features_df, self.patient_name or "환자", self)
            dialog.exec()

    def apply_edits(self):
        if self.mask_vol is None:
            self.log("❌ 적용할 마스크가 없습니다.")
            return

        edit_path = self.edit_item.get_merged_path()
        if edit_path.isEmpty():
            self.log("❌ 적용할 편집 내용이 없습니다.")
            return

        s = self.slice_slider.value()

        # Get current slice mask
        if self.current_view == "axial":
            current_mask = self.mask_vol[:, :, s].copy()
        elif self.current_view == "coronal":
            current_mask = self.mask_vol[:, s, :].copy()
        else:
            current_mask = self.mask_vol[s, :, :].copy()

        # Apply rotation and flip to match display
        display_mask = np.rot90(current_mask)
        display_mask = np.fliplr(display_mask)

        # Convert path to mask
        edit_mask = path_to_mask(edit_path, display_mask.shape)

        # Apply edit
        edited_mask = apply_edit_to_mask(display_mask, edit_mask, self.edit_item.mode)

        # Reverse transformations
        edited_mask = np.fliplr(edited_mask)
        edited_mask = np.rot90(edited_mask, k=-1)

        # Update mask volume
        if self.current_view == "axial":
            self.mask_vol[:, :, s] = edited_mask
        elif self.current_view == "coronal":
            self.mask_vol[:, s, :] = edited_mask
        else:
            self.mask_vol[s, :, :] = edited_mask

        self.edit_item.clear()
        self.update_slice_view()
        self.log("✅ 편집 적용 완료")

    # ----------------------------------------------------------
    def update_slice_view(self):
        if self.nifti_vol is None:
            return

        s = self.slice_slider.value()

        if self.current_view == "axial":
            img = np.rot90(self.nifti_vol[:, :, s])
            mask = np.rot90(self.mask_vol[:, :, s])
        elif self.current_view == "coronal":
            img = np.rot90(self.nifti_vol[:, s, :])
            mask = np.rot90(self.mask_vol[:, s, :])
        else:
            img = np.rot90(self.nifti_vol[s, :, :])
            mask = np.rot90(self.mask_vol[s, :, :])
        
        img = np.fliplr(img)
        mask = np.fliplr(mask)
        
        mask_edge = np.logical_xor(mask > 0, binary_erosion(mask > 0))

        # Normalize
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img_uint8 = (img_norm * 255).astype(np.uint8)

        overlay = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
        overlay[mask_edge] = [255, 0, 0]

        h, w, _ = overlay.shape
        qimg = QImage(overlay.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        pixmap = pixmap.scaled(
            int(w * self.zoom_factor), int(h * self.zoom_factor),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.viewer.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

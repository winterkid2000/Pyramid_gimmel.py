import sys
import os
import numpy as np
import nibabel as nib
import traceback
from scipy.ndimage import binary_erosion

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QFileDialog,
    QSlider, QPlainTextEdit, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPathItem, QRadioButton, QButtonGroup,
    QDialog
)
from PySide6.QtCore import Qt, Signal, QObject, QThread, QPointF
from PySide6.QtGui import (
    QPixmap, QImage, QPainterPath, QPen, QBrush, QPainter
)
import multiprocessing
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

from ras_converter import dicom_to_nifti_ras
from totalsegmentation import run_TS
from patient_information_collection import collect_patient_information
from radiomics_extr import extract_radiomics
from testor1 import predict_with_model

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import json
from typing import Optional, Dict, List
import traceback
from matplotlib.patches import Patch
import tempfile

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--prevent-loop":
        sys.exit(0)
    app = QApplication(sys.argv)
    w = RadiomicsAnalyzer()
    w.show()
    sys.exit(app.exec())

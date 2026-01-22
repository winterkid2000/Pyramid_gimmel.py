import pandas as pd
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------
#  Radiomics Feature Column List (이 순서대로 들어와야 함)
# --------------------------------------------------------
FEATURE_COLS = [
    'original_shape_Sphericity', 'original_glszm_GrayLevelNonUniformity',
    'original_glszm_SmallAreaEmphasis', 'wavelet-LHL_firstorder_Mean',
    'wavelet-LHL_glcm_Imc2', 'wavelet-LHH_firstorder_Mean',
    'wavelet-LHH_glszm_ZonePercentage', 'wavelet-HLH_firstorder_Mean',
    'wavelet-HHH_glcm_Imc1', 'wavelet-HHH_glcm_Imc2',
    'wavelet-HHH_glcm_InverseVariance',
    'wavelet-HHH_glszm_LowGrayLevelZoneEmphasis',
    'wavelet-HHH_glszm_SmallAreaLowGrayLevelEmphasis',
    'wavelet-LLL_firstorder_Uniformity',
    'wavelet-LLL_glszm_SizeZoneNonUniformityNormalized',
    'log-sigma-1-0-mm-3D_glszm_SmallAreaEmphasis',
    'log-sigma-3-0-mm-3D_glcm_Correlation',
    'log-sigma-3-0-mm-3D_glcm_Imc1',
    'log-sigma-4-0-mm-3D_glcm_Imc1',
    'log-sigma-4-0-mm-3D_glcm_MaximumProbability',
    'log-sigma-4-0-mm-3D_gldm_DependenceNonUniformityNormalized',
    'log-sigma-5-0-mm-3D_firstorder_Maximum',
    'log-sigma-5-0-mm-3D_glcm_Id',
    'log-sigma-5-0-mm-3D_glcm_InverseVariance',
    'log-sigma-5-0-mm-3D_glrlm_RunLengthNonUniformity',
    'log-sigma-5-0-mm-3D_glszm_SizeZoneNonUniformityNormalized',
    'log-sigma-5-0-mm-3D_gldm_DependenceNonUniformityNormalized',
    'square_glszm_GrayLevelNonUniformity',
    'square_glszm_SizeZoneNonUniformity', 'square_glszm_ZoneVariance',
    'squareroot_glszm_SizeZoneNonUniformityNormalized',
    'logarithm_glcm_Imc2',
    'logarithm_glszm_LargeAreaHighGrayLevelEmphasis',
    'logarithm_glszm_SizeZoneNonUniformityNormalized',
    'logarithm_glszm_ZoneEntropy', 'gradient_glcm_Imc1',
    'lbp-3D-m1_firstorder_Mean', 'lbp-3D-m2_firstorder_Kurtosis',
    'lbp-3D-k_firstorder_Maximum',
    'lbp-3D-k_firstorder_RobustMeanAbsoluteDeviation',
    'lbp-3D-k_ngtdm_Busyness'
]

# =========================================================
#   1) FTTransformer (최종 확정된 Optuna 파라미터 버전)
# =========================================================
class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4,
                 num_layers=4, dropout=0.229281172808302):
        super().__init__()

        self.value_embed = nn.Linear(1, d_model)
        self.feature_embed = nn.Parameter(torch.randn(input_dim, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.value_embed(x) + self.feature_embed

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)
        x = self.norm(x)

        return self.classifier(x[:, 0]).squeeze(1)


# =========================================================
#   2) Prediction Function (GUI에서 사용)
# =========================================================
def predict_with_model(df, name, model_path, scaler_path,
                       threshold=0.5, log_callback=None):

    try:
        if log_callback:
            log_callback("노스트라사무스 집중 중...")

        # ---------------------------
        #  Missing Feature Check
        # ---------------------------
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Radiomics feature 누락: {missing}")

        # ---------------------------
        #  Load model + scaler
        # ---------------------------
        model = FTTransformer(
            input_dim=len(FEATURE_COLS),
            d_model=64,
            num_heads=4,
            num_layers=4,
            dropout=0.229281172808302
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # ---------------------------
        #  Preprocess
        # ---------------------------
        X = df[FEATURE_COLS].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # ---------------------------
        #  Predict (stable sigmoid)
        # ---------------------------
        with torch.no_grad():
            logits = model(X_tensor).cpu().numpy()

        logits = np.squeeze(logits)
        probs = 1 / (1 + np.exp(-logits))  # stable sigmoid

        # ---------------------------
        #  Single case
        # ---------------------------
        if np.isscalar(probs):
            prob = float(probs)
            pred = "비정상" if prob >= threshold else "정상"

            df["Probability"] = [prob]
            df["Prediction"] = [pred]

            if log_callback:
                log_callback(f"환자 {name}의 비정상 예측 결과는 {prob*100:.2f}% 확률로 {pred}입니다.")

        else:
            # ---------------------------
            #  Multiple cases
            # ---------------------------
            df["Probability"] = probs
            df["Prediction"] = [
                "비정상" if p >= threshold else "정상" for p in probs
            ]

            if log_callback:
                log_callback(
                    f"첫 환자 결과: {probs[0]*100:.2f}% "
                    f"({'비정상' if probs[0] >= threshold else '정상'})"
                )

        return df

    except Exception as e:
        if log_callback:
            log_callback(f"돌팔이였습니다...: {str(e)}")
        return None

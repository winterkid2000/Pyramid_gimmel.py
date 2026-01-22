import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import yaml

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
    'logarithm_glszm_ZoneEntropy',
    'gradient_glcm_Imc1', 'lbp-3D-m1_firstorder_Mean',
    'lbp-3D-m2_firstorder_Kurtosis', 'lbp-3D-k_firstorder_Maximum',
    'lbp-3D-k_firstorder_RobustMeanAbsoluteDeviation',
    'lbp-3D-k_ngtdm_Busyness'
]


def build_extractor(yaml_path):

    if yaml_path and os.path.exists(yaml_path):

        return featureextractor.RadiomicsFeatureExtractor(yaml_path)

    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.enableAllImageTypes()
    ext.enableAllFeatures()
    return ext


def extract_radiomics(img_nii, mask_nii, yaml_path):

    extractor = build_extractor(yaml_path)

    img = sitk.ReadImage(img_nii)
    mask = sitk.ReadImage(mask_nii)

    raw_features = extractor.execute(img, mask)


    selected = {}

    for feat in FEATURE_COLS:
        if feat in raw_features:
            val = raw_features[feat]

            # numpy 타입 처리
            if isinstance(val, (np.generic, np.ndarray)):
                selected[feat] = float(val)
            else:
                selected[feat] = val

        else:
            selected[feat] = np.nan

    df = pd.DataFrame([selected])
    return df

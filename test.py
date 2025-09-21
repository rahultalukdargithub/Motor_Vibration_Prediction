from fastapi import UploadFile, File ,Form
from io import BytesIO
import os
import numpy as np
import scipy.io
from joblib import load
from utils import combine_axes_normalize, emd_denoise, extract_F4

MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "default_model.joblib")

async def handle_test_signal(
    file: UploadFile = File(...), 
    motor_id: str = Form("motor_default")   # client specifies motor_id
):
    # Resolve model path
    model_path = os.path.join(MODELS_DIR, f"{motor_id}_model.joblib")
    used_fallback = False
    if not os.path.exists(model_path):
        print("########## Loading the Default Model ##########")
        model_path = DEFAULT_MODEL_PATH   # fallback to default
        used_fallback = True

    if not os.path.exists(model_path):
        return {"error": "No trained model available."}

    model = load(model_path)

    # Read .mat
    contents = await file.read()
    mat_data = scipy.io.loadmat(BytesIO(contents))
    if 'H' not in mat_data:
        return {"error": "Uploaded .mat missing expected 'H' key"}

    H = np.squeeze(np.array(mat_data['H'])).astype(float)
    x, y, z = H[:,0], H[:,1], H[:,2]

    s = combine_axes_normalize(x, y, z)
    s_d = emd_denoise(s)
    feats = extract_F4(s_d).reshape(1, -1)

    raw_score = model.decision_function(feats).ravel()[0]
    fault_degree = 1 / (1 + np.exp(raw_score))
    pred = model.predict(feats)[0]
    pred_label = 0 if pred == 1 else 1

    result = {
        "motor_id": motor_id if not used_fallback else f"{motor_id} (default fallback used)",
        "fault_degree": float(fault_degree)*100,
        "predicted_label": "Faulty" if int(pred_label)==1 else "Healthy",
        "raw_decision_score": float(raw_score)
    }
    return result


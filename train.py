# train.py
from fastapi import UploadFile, File , Form
from io import BytesIO
import zipfile
import tempfile
from joblib import dump
from utils import load_folder_mat, build_dataset_features, tune_svm_rbf
import os


MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "default_model.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)


async def handle_train_model(
    file: UploadFile = File(...),
    motor_id: str = Form("motor_default")   # client specifies motor_id
):
    contents = await file.read()
    memory_file = BytesIO(contents)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(memory_file) as archive:
            archive.extractall(tmpdir)

        extracted_folders = [f for f in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, f))]
        print("#################################")
        print(extracted_folders)
        print("#################################")
        if not extracted_folders:
            raise ValueError("No folder found inside the ZIP file")
        
        folder_path = os.path.join(tmpdir, extracted_folders[0])
        Xx_h, Xy_h, Xz_h, y_h = load_folder_mat(folder_path, label=0)
        feats_df_h = build_dataset_features(Xx_h, Xy_h, Xz_h)
        X_feats_h = feats_df_h.values

        # Train tuned one-class SVM
        best_model = tune_svm_rbf(X_feats_h, y_h, random_state=42)

        # Save per-motor model
        model_path = os.path.join(MODELS_DIR, f"{motor_id}_model.joblib")
        dump(best_model, model_path)

    return {"message": f"Model trained and saved for {motor_id}.", "model_path": model_path}




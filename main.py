from fastapi import FastAPI, UploadFile, File, Form
from test import handle_test_signal
from train import handle_train_model

app = FastAPI()

@app.post("/test_signal/")
async def test_signal(
    file: UploadFile = File(...),
    motor_id: str = Form("motor_default")
):
    result = await handle_test_signal(file=file, motor_id=motor_id)
    return result

@app.post("/train_model/")
async def train_model(
    file: UploadFile = File(...),
    motor_id: str = Form("motor_default")
):
    result = await handle_train_model(file=file, motor_id=motor_id)
    return result

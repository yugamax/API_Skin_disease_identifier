import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import uvicorn
from PIL import Image
import io
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


yugask = load_model(r"model/skin_disease_model.keras")

class_list = ["Acne and Rosacea", "Athelete foots", "Chickenpox", "Give proper skin image! (zoom/focus on the skin)", "Melanocytic nevus", "Melanoma", "Shingles", "Squamous cell carcinoma", "Tinea Ringworm Candidiasis", "Vascular lesion" , "You have a Clear skin"]

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    await asyncio.sleep(0.1)
    return {"message": "server is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        imgb = await file.read()
        img = Image.open(io.BytesIO(imgb)).convert("RGB")
        img = img.resize((224 , 224))  
        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        pred = yugask.predict(img_arr)
        pred_ind = np.argmax(pred, axis=1)[0]
        pred_conf = pred[0][pred_ind]
        
        if float(pred_conf) < 0.6:
            if float(pred_conf) < 0.5:
                return {"predicted_class": class_list[3], "confidence": (float(1)-float(pred_conf)) * 100}
            else:
                return {"predicted_class": class_list[3], "confidence": float(pred_conf) * 100}
        else:
            return {"predicted_class": class_list[pred_ind], "confidence": float(pred_conf) * 100}
                
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import uvicorn
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


yugask = load_model(r"model/skin_disease_model.keras")

CLASS_LABELS = ["Class_Acne and Rosacea Photos", "ClasAthelete foots_2", "Chickenpox", "Give proper skin image! (zoom on the skin)", "Melanocytic nevus", "Melanoma", "Shingles", "Squamous cell carcinoma", "Tinea Ringworm Candidiasis", "Vascular lesion" , "You have a Clear skin"]

@app.get("/ping")
async def ping():
    return {"message": "Server is running!"}

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
        
        return {"predicted_class": CLASS_LABELS[pred_ind], "confidence": float(pred_conf) * 100}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
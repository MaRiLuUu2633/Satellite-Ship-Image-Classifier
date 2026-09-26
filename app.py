from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
from PIL import Image
import numpy as np
import io
import os
import time
import shutil
import joblib
from skimage.feature import hog, local_binary_pattern
import warnings

warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
IMG_SIZE = (80, 80)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_BINS = 26

model_path = 'model/best_ship_ml.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

def preprocess_image(image):
    img = image.convert('RGB')
    img_resized = img.resize(IMG_SIZE)
    img_np = np.array(img_resized)
    
    # 1. HOG Features (Grayscale)
    img_gray = img_resized.convert('L')
    hog_features = hog(
        np.array(img_gray),
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    # 2. RGB Color Histograms
    total_pixels = IMG_SIZE[0] * IMG_SIZE[1]
    hist_r, _ = np.histogram(img_np[:,:,0], bins=32, range=(0, 256))
    hist_g, _ = np.histogram(img_np[:,:,1], bins=32, range=(0, 256))
    hist_b, _ = np.histogram(img_np[:,:,2], bins=32, range=(0, 256))
    hist_rgb = np.concatenate([hist_r, hist_g, hist_b]) / total_pixels
    
    # 3. HSV Color Histograms
    img_hsv = img_resized.convert('HSV')
    np_hsv = np.array(img_hsv)
    hist_h, _ = np.histogram(np_hsv[:,:,0], bins=32, range=(0, 256))
    hist_s, _ = np.histogram(np_hsv[:,:,1], bins=32, range=(0, 256))
    hist_v, _ = np.histogram(np_hsv[:,:,2], bins=32, range=(0, 256))
    hist_hsv = np.concatenate([hist_h, hist_s, hist_v]) / total_pixels
    
    # 4. Color Moments (RGB + HSV)
    moments_rgb = [
        np.mean(img_np[:,:,0]), np.std(img_np[:,:,0]),
        np.mean(img_np[:,:,1]), np.std(img_np[:,:,1]),
        np.mean(img_np[:,:,2]), np.std(img_np[:,:,2])
    ]
    moments_hsv = [
        np.mean(np_hsv[:,:,0]), np.std(np_hsv[:,:,0]),
        np.mean(np_hsv[:,:,1]), np.std(np_hsv[:,:,1]),
        np.mean(np_hsv[:,:,2]), np.std(np_hsv[:,:,2])
    ]
    moments = np.array(moments_rgb + moments_hsv) / 255.0
    
    # 5. Local Binary Pattern (LBP)
    lbp = local_binary_pattern(np.array(img_gray), LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
    lbp_hist = lbp_hist / lbp_hist.sum()
    
    features = np.concatenate([hog_features, hist_rgb, hist_hsv, moments, lbp_hist])
    return features.reshape(1, -1)

@app.get("/status")
def status():
    return {"status": "running", "engine": "classical_ml", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    features = preprocess_image(image)
    
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
        
    prediction = "ship" if pred == 1 else "no_ship"
    
    return {"prediction": prediction, "confidence": prob if pred == 1 else 1 - prob}

TEST_DAY_DIR = 'dataset/test_day_images'
os.makedirs(TEST_DAY_DIR, exist_ok=True)

@app.post("/upload_test_images")
async def upload_test_images(files: List[UploadFile] = File(...)):
    # Clear directory to start fresh for a new test day
    for f in os.listdir(TEST_DAY_DIR):
        os.remove(os.path.join(TEST_DAY_DIR, f))
        
    saved_files = []
    for file in files:
        file_path = os.path.join(TEST_DAY_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
        
    return {"filenames": saved_files}

# Add route to serve the test day images so the UI can display them
app.mount("/test_images", StaticFiles(directory=TEST_DAY_DIR), name="test_images")

class EvaluationLabels(BaseModel):
    labels: Dict[str, int]

@app.post("/evaluate_test_day")
def evaluate_test_day(data: EvaluationLabels):
    if model is None:
        return {"error": "Model not loaded"}
        
    correct = 0
    total = 0
    true_labels = []
    predictions = []
    results_detail = []
    
    start_time = time.time()
    
    for filename, true_label in data.labels.items():
        file_path = os.path.join(TEST_DAY_DIR, filename)
        if not os.path.exists(file_path):
            continue
            
        img = Image.open(file_path)
        features = preprocess_image(img)
        
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
            
        true_labels.append(true_label)
        predictions.append(pred)
        
        if pred == true_label:
            correct += 1
        total += 1
        
        results_detail.append({
            "filename": filename,
            "true_label": true_label,
            "prediction": int(pred),
            "confidence": float(prob if pred == 1 else 1 - prob)
        })
        
    end_time = time.time()
    
    accuracy = correct / total if total > 0 else 0
    tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        },
        "time_taken": end_time - start_time,
        "total_images": total,
        "results_detail": results_detail
    }

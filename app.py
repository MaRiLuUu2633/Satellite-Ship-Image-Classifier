# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ShipCNN(nn.Module):
    def __init__(self):
        super(ShipCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ShipCNN().to(DEVICE)
model_path = 'model/best_ship_cnn.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

val_transform = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/status")
def status():
    return {"status": "running", "device": str(DEVICE), "model_loaded": os.path.exists(model_path)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor).squeeze()
        prob = torch.sigmoid(output).item()
        
    # prob is float. if output is 0 dim, item() gets it.
    prediction = "ship" if prob >= 0.5 else "no_ship"
    
    return {"prediction": prediction, "confidence": prob if prediction == "ship" else 1 - prob}

@app.get("/evaluate")
def evaluate():
    blind_test_dir = 'dataset/blind_test'
    if not os.path.exists(blind_test_dir):
        return {"error": "Blind test set not found"}
        
    correct = 0
    total = 0
    
    true_labels = []
    predictions = []
    
    start_time = time.time()
    
    # Evaluate ships
    ship_dir = os.path.join(blind_test_dir, 'ship')
    if os.path.exists(ship_dir):
        for f in os.listdir(ship_dir):
            if f.endswith('.png'):
                img = Image.open(os.path.join(ship_dir, f)).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(tensor).squeeze()
                    prob = torch.sigmoid(output).item()
                
                pred = 1 if prob >= 0.5 else 0
                true_labels.append(1)
                predictions.append(pred)
                if pred == 1:
                    correct += 1
                total += 1

    # Evaluate no ships
    no_ship_dir = os.path.join(blind_test_dir, 'no_ship')
    if os.path.exists(no_ship_dir):
        for f in os.listdir(no_ship_dir):
            if f.endswith('.png'):
                img = Image.open(os.path.join(no_ship_dir, f)).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(tensor).squeeze()
                    prob = torch.sigmoid(output).item()
                
                pred = 1 if prob >= 0.5 else 0
                true_labels.append(0)
                predictions.append(pred)
                if pred == 0:
                    correct += 1
                total += 1
                
    end_time = time.time()
    
    accuracy = correct / total if total > 0 else 0
    
    # Calculate Precision and Recall
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
        "total_images": total
    }

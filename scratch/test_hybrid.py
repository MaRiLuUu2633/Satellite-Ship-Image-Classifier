import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = 'dataset/train_val'
BLIND_TEST_DIR = 'dataset/blind_test'
IMG_SIZE = (80, 80)

# Setup device & CNN feature extractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pretrained ResNet18 feature extractor
weights = models.ResNet18_Weights.DEFAULT
base_model = models.resnet18(weights=weights)
feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) # Output shape: (N, 512, 1, 1)
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_deep_ml_features(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert('RGB')
    else:
        img = img_path_or_pil.convert('RGB')
        
    tensor_img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(tensor_img).squeeze().cpu().numpy() # 512-dim vector
        
    # Also extract color & HOG features to boost exactness
    img_np = np.array(img.resize(IMG_SIZE))
    total_pixels = IMG_SIZE[0] * IMG_SIZE[1]
    hist_r, _ = np.histogram(img_np[:,:,0], bins=16, range=(0, 256))
    hist_g, _ = np.histogram(img_np[:,:,1], bins=16, range=(0, 256))
    hist_b, _ = np.histogram(img_np[:,:,2], bins=16, range=(0, 256))
    hist_rgb = np.concatenate([hist_r, hist_g, hist_b]) / total_pixels
    
    combined = np.concatenate([feat, hist_rgb])
    return combined

def load_dataset(root_dir):
    print(f"Extracting features from {root_dir}...")
    X, y = [], []
    for cls, label in [('ship', 1), ('no_ship', 0)]:
        d = os.path.join(root_dir, cls)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.png') or f.endswith('.jpg'):
                    X.append(extract_deep_ml_features(os.path.join(d, f)))
                    y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    start = time.time()
    X_train, y_train = load_dataset(DATA_DIR)
    X_test, y_test = load_dataset(BLIND_TEST_DIR)
    print(f"Features extracted in {time.time()-start:.2f}s. Feature shape: {X_train.shape}")
    
    # ML Ensemble
    rf = RandomForestClassifier(n_estimators=500, min_samples_split=2, class_weight='balanced', n_jobs=-1, random_state=42)
    et = ExtraTreesClassifier(n_estimators=500, min_samples_split=2, class_weight='balanced', n_jobs=-1, random_state=42)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=0.1, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    
    ensemble = VotingClassifier(estimators=[('rf', rf), ('et', et), ('hgb', hgb), ('mlp', mlp)], voting='soft')
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ensemble)
    ])
    
    print("Fitting Machine Learning Ensemble...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nHybrid ML Blind Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Goal Achieved (> 98%): {'YES' if acc >= 0.98 else 'NO'}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["No Ship", "Ship"]))

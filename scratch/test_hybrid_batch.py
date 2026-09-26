import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for cls, label in [('ship', 1), ('no_ship', 0)]:
            d = os.path.join(root_dir, cls)
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(d, f), label))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(base_model.children())[:-1]).to(device)
feature_extractor.eval()

def extract_features_batched(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    features_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            features_list.append(feats)
            labels_list.append(lbls.numpy())
    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y

if __name__ == "__main__":
    t0 = time.time()
    train_ds = SatelliteDataset(DATA_DIR, transform=transform)
    test_ds = SatelliteDataset(BLIND_TEST_DIR, transform=transform)
    
    print(f"Loaded datasets. Extracting features for {len(train_ds)} train & {len(test_ds)} test images...")
    X_train, y_train = extract_features_batched(train_ds)
    X_test, y_test = extract_features_batched(test_ds)
    print(f"Features extracted in {time.time()-t0:.2f} seconds. Shape: {X_train.shape}")
    
    # Train high-capacity Ensemble ML Classifier
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2, class_weight='balanced', n_jobs=-1, random_state=42)
    et = ExtraTreesClassifier(n_estimators=400, max_depth=None, min_samples_split=2, class_weight='balanced', n_jobs=-1, random_state=42)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=0.1, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    
    ensemble = VotingClassifier(estimators=[('rf', rf), ('et', et), ('hgb', hgb), ('mlp', mlp)], voting='soft')
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ensemble)
    ])
    
    print("Training Machine Learning Ensemble...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n==========================================")
    print(f"HYBRID ML BLIND TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(f"GOAL ACHIEVED (>98%): {'YES' if acc >= 0.98 else 'NO'}")
    print(f"==========================================\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["No Ship", "Ship"]))
    
    os.makedirs('model', exist_ok=True)
    # Save the pipeline and feature extraction reference
    joblib.dump(pipeline, 'model/best_ship_ml.pkl')
    print("\nSaved model pipeline to model/best_ship_ml.pkl")

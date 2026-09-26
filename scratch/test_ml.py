import os
import time
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = 'dataset/train_val'
BLIND_TEST_DIR = 'dataset/blind_test'
IMG_SIZE = (80, 80)

def extract_advanced_ml_features(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert('RGB')
    else:
        img = img_path_or_pil.convert('RGB')
        
    img_resized = img.resize(IMG_SIZE)
    img_np = np.array(img_resized)
    img_gray = np.array(img_resized.convert('L'))
    
    # 1. Multi-scale HOG Features
    hog1 = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    hog2 = hog(img_gray, orientations=12, pixels_per_cell=(10, 10), cells_per_block=(2, 2), feature_vector=True)
    
    # 2. Color Histograms (RGB + HSV + LAB)
    total_pixels = IMG_SIZE[0] * IMG_SIZE[1]
    hist_r, _ = np.histogram(img_np[:,:,0], bins=32, range=(0, 256))
    hist_g, _ = np.histogram(img_np[:,:,1], bins=32, range=(0, 256))
    hist_b, _ = np.histogram(img_np[:,:,2], bins=32, range=(0, 256))
    hist_rgb = np.concatenate([hist_r, hist_g, hist_b]) / total_pixels
    
    img_hsv = np.array(img_resized.convert('HSV'))
    hist_h, _ = np.histogram(img_hsv[:,:,0], bins=32, range=(0, 256))
    hist_s, _ = np.histogram(img_hsv[:,:,1], bins=32, range=(0, 256))
    hist_v, _ = np.histogram(img_hsv[:,:,2], bins=32, range=(0, 256))
    hist_hsv = np.concatenate([hist_h, hist_s, hist_v]) / total_pixels
    
    # 3. Spatial Color Moments (Global + 2x2 Quadrants)
    quadrants = [
        img_np[0:40, 0:40, :], img_np[0:40, 40:80, :],
        img_np[40:80, 0:40, :], img_np[40:80, 40:80, :]
    ]
    spatial_moments = []
    for q in quadrants:
        for c in range(3):
            spatial_moments.extend([np.mean(q[:,:,c]), np.std(q[:,:,c])])
    spatial_moments = np.array(spatial_moments) / 255.0
    
    # 4. Multi-Radius LBP
    lbp1 = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    hist_lbp1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10))
    hist_lbp1 = hist_lbp1 / hist_lbp1.sum()
    
    lbp2 = local_binary_pattern(img_gray, P=24, R=3, method='uniform')
    hist_lbp2, _ = np.histogram(lbp2.ravel(), bins=26, range=(0, 26))
    hist_lbp2 = hist_lbp2 / hist_lbp2.sum()
    
    # 5. GLCM Texture Features
    glcm = graycomatrix(img_gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    glcm_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    
    features = np.concatenate([hog1, hog2, hist_rgb, hist_hsv, spatial_moments, hist_lbp1, hist_lbp2, glcm_features])
    return features

def load_data(root_dir):
    print(f"Extracting features from {root_dir}...")
    X, y = [], []
    for cls, label in [('ship', 1), ('no_ship', 0)]:
        d = os.path.join(root_dir, cls)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.png') or f.endswith('.jpg'):
                    X.append(extract_advanced_ml_features(os.path.join(d, f)))
                    y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    start = time.time()
    X_train, y_train = load_data(DATA_DIR)
    X_test, y_test = load_data(BLIND_TEST_DIR)
    print(f"Features extracted in {time.time()-start:.2f}s. Dim: {X_train.shape[1]}")
    
    rf = RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_split=2, class_weight='balanced', n_jobs=-1, random_state=42)
    et = ExtraTreesClassifier(n_estimators=800, max_depth=None, min_samples_split=2, class_weight='balanced', n_jobs=-1, random_state=42)
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=0.5, random_state=42)
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
    print(f"\nMachine Learning Blind Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["No Ship", "Ship"]))

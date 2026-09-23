import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ----------------- Configuration -----------------
DATA_DIR = 'dataset/train_val'
BLIND_TEST_DIR = 'dataset/blind_test'
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
KFOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- Dataset -----------------
class ShipDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        
        # Load ship images
        ship_dir = os.path.join(root_dir, 'ship')
        if os.path.exists(ship_dir):
            for f in os.listdir(ship_dir):
                if f.endswith('.png'):
                    self.image_paths.append(os.path.join(ship_dir, f))
                    self.labels.append(1)
                    
        # Load no_ship images
        no_ship_dir = os.path.join(root_dir, 'no_ship')
        if os.path.exists(no_ship_dir):
            for f in os.listdir(no_ship_dir):
                if f.endswith('.png'):
                    self.image_paths.append(os.path.join(no_ship_dir, f))
                    self.labels.append(0)
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((80, 80))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_np = (img_np - mean) / std
        
        return torch.tensor(img_np, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# ----------------- Architecture -----------------
class VisualDescriptorExtractor(nn.Module):
    def __init__(self):
        super(VisualDescriptorExtractor, self).__init__()
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
        
    def forward(self, x):
        return self.features(x)

class ShipClassifier(nn.Module):
    def __init__(self):
        super(ShipClassifier, self).__init__()
        self.descriptors = VisualDescriptorExtractor()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        desc = self.descriptors(x)
        return self.classifier(desc)

def train_and_validate():
    print("Loading Dataset...")
    full_dataset = ShipDataset(DATA_DIR)
    n_samples = len(full_dataset)
    print(f"Loaded {n_samples} images.")
    
    # Manual KFold to avoid sklearn crash
    indices = np.random.RandomState(42).permutation(n_samples)
    fold_sizes = np.full(KFOLDS, n_samples // KFOLDS, dtype=int)
    fold_sizes[:n_samples % KFOLDS] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    
    fold_results = []
    best_overall_acc = 0.0
    best_model_state = None
    
    for fold in range(KFOLDS):
        print(f"\n--- Fold {fold+1}/{KFOLDS} ---")
        val_idx = folds[fold]
        train_idx = np.hstack([folds[i] for i in range(KFOLDS) if i != fold])
        
        train_sub = torch.utils.data.Subset(full_dataset, train_idx)
        val_sub = torch.utils.data.Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        labels = [full_dataset.labels[i] for i in train_idx]
        pos_weight = torch.tensor([(len(labels) - sum(labels)) / max(sum(labels), 1)], dtype=torch.float32).to(DEVICE)
        
        model = ShipClassifier().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_fold_acc = 0.0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for images, lbls in train_loader:
                images, lbls = images.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, lbls in val_loader:
                    images, lbls = images.to(DEVICE), lbls.to(DEVICE)
                    outputs = model(images).squeeze()
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
                    
            val_acc = correct / total
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                
        fold_results.append(best_fold_acc)
        print(f"Best Fold {fold+1} Accuracy: {best_fold_acc:.4f}")
        
        if best_fold_acc > best_overall_acc:
            best_overall_acc = best_fold_acc
            best_model_state = model.state_dict()
            
    print(f"\n--- Cross Validation Results ---")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"Mean CV Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    
    os.makedirs('model', exist_ok=True)
    torch.save(best_model_state, 'model/best_ship_cnn.pth')
    print("Saved best model to model/best_ship_cnn.pth")
    
    # Evaluate on blind test manually
    print("\nEvaluating on Blind Test Set...")
    test_dataset = ShipDataset(BLIND_TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(test_dataset) > 0:
        best_model = ShipClassifier().to(DEVICE)
        best_model.load_state_dict(best_model_state)
        best_model.eval()
        
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        with torch.no_grad():
            for images, lbls in test_loader:
                images = images.to(DEVICE)
                outputs = best_model(images).squeeze()
                preds = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())
        end_time = time.time()
        
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        test_acc = correct / len(all_labels)
        print(f"Blind Test Accuracy: {test_acc:.4f}")
        print(f"Goal Achieved: {'YES' if test_acc > 0.98 else 'NO'}")
        
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
        
        print("\nConfusion Matrix:")
        print(f"[[{tn} {fp}]\n [{fn} {tp}]]")
        print(f"Time taken for evaluation: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    train_and_validate()

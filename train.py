import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# EXPERIMENTO v1.5 - Fine-tuning end-to-end
# Reemplaza el enfoque anterior (features congeladas + ensamble sklearn)
# por entrenamiento end-to-end de layer3/layer4 + cabeza de clasificacion.
# IMG_SIZE se mantiene en 80x80 (requerimiento fijo).
# Objetivo: Accuracy > 98.0% (no garantizado, depende tambien de los datos)
# ==========================================================

DATA_DIR = 'dataset/train_val'
BLIND_TEST_DIR = 'dataset/blind_test'
MODEL_PATH = 'model/best_ship_finetuned.pt'

IMG_SIZE = (80, 80)   # SE MANTIENE, es requerimiento
BATCH_SIZE = 64
VAL_SPLIT = 0.15       # se separa una porcion de train_val para validar/early-stopping
NUM_EPOCHS = 40
PATIENCE = 8           # epochs sin mejora en val_acc antes de detener
SEED = 42

HEAD_LR = 1e-3         # LR mas alto: la cabeza nueva parte de pesos aleatorios
BACKBONE_LR = 1e-4     # LR mas bajo: layer3/layer4 ya vienen pre-entrenados
WEIGHT_DECAY = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Vistas para Test-Time Augmentation (TTA): se promedian las probabilidades
# de estas 4 vistas sobre cada imagen del blind test, sin volver a entrenar nada.
tta_transforms = [
    eval_transform,
    transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)
    ]),
    transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)
    ]),
    transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.RandomRotation((90, 90)),
        transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)
    ]),
]


def list_samples(root_dir):
    samples = []
    for cls, label in [('ship', 1), ('no_ship', 0)]:
        d = os.path.join(root_dir, cls)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((os.path.join(d, f), label))
    return samples


class SatelliteDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class ResNetFineTune(nn.Module):
    """
    Backbone ResNet34 adaptado a imagenes pequenas (80x80):
      - maxpool inicial eliminado -> preserva resolucion espacial
      - conv1/bn1/layer1/layer2 CONGELADOS -> features genericos de bajo nivel,
        no hace falta reaprenderlos y congelarlos evita overfitting con pocos datos
      - layer3/layer4 DESCONGELADOS -> se afinan para este dataset y resolucion
      - cabeza de clasificacion pequena con dropout para regularizar
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity()
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
            for p in module.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return self.head(x)


def build_optimizer(model):
    head_params = list(model.head.parameters())
    backbone_params = list(model.layer3.parameters()) + list(model.layer4.parameters())
    return optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': HEAD_LR}
    ], weight_decay=WEIGHT_DECAY)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def predict_with_tta(model, samples):
    """Promedia probabilidades sobre varias vistas (TTA) para cada imagen de test."""
    model.eval()
    all_probs = None
    all_labels = np.array([lbl for _, lbl in samples])

    for t in tta_transforms:
        ds = SatelliteDataset(samples, transform=t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        probs_list = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        probs_view = np.vstack(probs_list)
        all_probs = probs_view if all_probs is None else all_probs + probs_view

    all_probs /= len(tta_transforms)
    preds = all_probs.argmax(axis=1)
    return preds, all_labels


def train_and_validate():
    print(f"Dispositivo: {device}")
    samples = list_samples(DATA_DIR)
    labels = [lbl for _, lbl in samples]

    train_samples, val_samples = train_test_split(
        samples, test_size=VAL_SPLIT, stratify=labels, random_state=SEED
    )
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = SatelliteDataset(train_samples, transform=train_transform)
    val_ds = SatelliteDataset(val_samples, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_labels_only = np.array([lbl for _, lbl in train_samples])
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=train_labels_only
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    model = ResNetFineTune().to(device)
    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping en epoch {epoch} (mejor val_acc={best_val_acc:.4f})")
                break

    print(f"Entrenamiento completo en {time.time()-t0:.2f}s. Mejor val_acc: {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # ------------------------------------------------------------
    # Evaluacion en blind test con Test-Time Augmentation (TTA)
    # ------------------------------------------------------------
    test_samples = list_samples(BLIND_TEST_DIR)
    print(f"\nEvaluando blind test ({len(test_samples)} imagenes) con TTA "
          f"({len(tta_transforms)} vistas)...")
    preds, y_test = predict_with_tta(model, test_samples)

    print("\n==================================================")
    print("EVALUACION EN BLIND TEST SET (Prueba a Ciegas)")
    print("==================================================")
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy final en Blind Test: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Objetivo alcanzado (> 98.0%): {'SI - LOGRADO' if acc >= 0.98 else 'NO'}")

    print("\nMatriz de Confusion:")
    print(confusion_matrix(y_test, preds))
    print("\nReporte de Clasificacion:")
    print(classification_report(y_test, preds, target_names=["No Ship", "Ship"]))


if __name__ == "__main__":
    train_and_validate()

# ==========================================================
# QUE CAMBIA RESPECTO A v1.4 (features congeladas + ensamble sklearn)
# ==========================================================
# v1.4 usaba ResNet como un extractor FIJO (sin gradientes) y todo el
# aprendizaje ocurria en clasificadores clasicos (RF, ExtraTrees, HGB, MLP)
# sobre esos features. Eso tiene un techo: si los features genericos de
# ImageNet no separan bien tus dos clases a 80x80, ningun clasificador
# clasico lo va a arreglar, por mas grande que sea el ensamble.
#
# v1.5 en cambio:
#
# 1) Fine-tuning real de layer3 y layer4:
#    Los pesos de estas capas SI se actualizan con tus datos (backprop). La
#    red literalmente reaprende que patrones visuales de una imagen de 80x80
#    distinguen "ship" de "no_ship", en vez de usar filtros aprendidos para
#    reconocer objetos de ImageNet a 224x224.
#
# 2) conv1/bn1/layer1/layer2 se mantienen CONGELADOS:
#    Las primeras capas de una CNN aprenden bordes, texturas y gradientes de
#    color, que son geneticos y utiles para casi cualquier tarea de vision.
#    Congelarlas reduce drasticamente el numero de parametros entrenables,
#    lo cual importa porque tu dataset probablemente no es enorme: menos
#    parametros entrenables = menor riesgo de sobreajuste.
#
# 3) Dos learning rates distintos (BACKBONE_LR=1e-4, HEAD_LR=1e-3):
#    La cabeza de clasificacion parte de pesos aleatorios y necesita moverse
#    mas rapido; layer3/layer4 ya estan cerca de un buen optimo (pre-entrenados)
#    y solo necesitan un ajuste fino, no una LR agresiva que los "rompa".
#
# 4) Split train/val real (85/15) + early stopping:
#    v1.4 no tenia validacion, entrenaba con TODO train_val sin forma de saber
#    si estaba sobreajustando hasta ver el blind test. Aqui se monitorea
#    val_acc epoch a epoch y se guarda el MEJOR checkpoint, no el ultimo.
#
# 5) Test-Time Augmentation (TTA) en el blind test:
#    Se predice el mismo lote de imagenes bajo 4 vistas (original, flip
#    horizontal, flip vertical, rotacion 90) y se promedian las
#    probabilidades antes de decidir la clase. Es una ganancia "gratis"
#    (no requiere reentrenar nada) que suele aportar entre 0.3% y 1.5% de
#    accuracy adicional en tareas donde la orientacion no importa, como es
#    el caso de imagenes satelitales.
#
# 6) class_weight balanceado via CrossEntropyLoss(weight=...):
#    Igual que en v1.4, pero ahora aplicado directamente en la funcion de
#    perdida de la red, no en un clasificador sklearn.
#
# SI DESPUES DE ESTO SIGUES SIN LLEGAR A 98%
# ==========================================================
# Es una senal de que el techo ya no esta en la arquitectura sino en los
# datos. Cosas a revisar, en orden de probabilidad de ayudar:
#   - Revisa que no haya imagenes mal etiquetadas en train_val o blind_test
#     (un pequeno % de labels incorrectos limita el accuracy maximo posible).
#   - Revisa el tamano real del dataset: fine-tuning con muy pocas imagenes
#     por clase (¿cientos en vez de miles?) tiene margen de mejora limitado
#     sin mas datos o mas augmentacion.
#   - Prueba descongelar tambien layer2 (mas capacidad, mas riesgo de
#     overfitting: solo si tienes datos suficientes).
#   - Sube NUM_EPOCHS/PATIENCE si el entrenamiento se corta antes de converger
#     (revisa la curva de val_acc en el log de cada epoch).
#   - Prueba una arquitectura distinta como backbone (EfficientNet-B0 suele
#     comportarse mejor que ResNet en imagenes pequenas).
# ==========================================================
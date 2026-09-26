# 🚢 Ship Detection AI

Sistema de clasificación binaria (**ship** / **no_ship**) sobre imágenes satelitales de 80×80 píxeles, con un pipeline de entrenamiento en PyTorch y una aplicación de escritorio (CustomTkinter) para inferencia local, sin conexión a internet.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `train_ships_v1.5_finetune.py` | Entrena el modelo: fine-tuning end-to-end de un ResNet34 adaptado a imágenes pequeñas, con validación, early stopping y evaluación en blind test con Test-Time Augmentation (TTA). |
| `ship_detection_ui_v1.5.py` | Aplicación de escritorio para clasificar una imagen individual o una carpeta completa, con métricas (accuracy, precisión, recall, matriz de confusión) cuando hay etiquetas reales disponibles. |
| `model/` | Carpeta donde se guarda el modelo entrenado (`best_ship_finetuned.pt`). No se versiona en git (ver `.gitignore`). |
| `dataset/` | Carpeta esperada con las imágenes de entrenamiento y prueba. No se versiona en git. |

## Arquitectura del modelo

- Backbone **ResNet34** pre-entrenado en ImageNet, adaptado para imágenes de 80×80:
  - Se elimina el `maxpool` inicial para preservar resolución espacial.
  - `conv1`, `bn1`, `layer1` y `layer2` quedan congelados (features genéricos de bajo nivel).
  - `layer3` y `layer4` se afinan (fine-tuning) específicamente para este dataset.
- Cabeza de clasificación propia con `Dropout` para regularizar.
- **Test-Time Augmentation (TTA)**: en evaluación se promedian las probabilidades sobre 4 vistas (original, flip horizontal, flip vertical, rotación 90°).

## Estructura de datos esperada

```
dataset/
├── train_val/
│   ├── ship/
│   │   ├── imagen1.png
│   │   └── ...
│   └── no_ship/
│       ├── imagen1.png
│       └── ...
└── blind_test/
    ├── ship/
    └── no_ship/
```

## Requisitos

- Python 3.9+
- PyTorch y torchvision
- scikit-learn
- Pillow
- CustomTkinter (solo para la UI de escritorio)

Instalación rápida:

```bash
pip install torch torchvision scikit-learn pillow customtkinter
```

> Si tienes GPU con CUDA, instala la versión de PyTorch correspondiente desde [pytorch.org](https://pytorch.org/get-started/locally/) para acelerar el entrenamiento.

## Uso

### 1. Entrenar el modelo

```bash
python train_ships_v1.5_finetune.py
```

Esto genera `model/best_ship_finetuned.pt` y muestra en consola la accuracy, matriz de confusión y reporte de clasificación sobre el blind test.

### 2. Ejecutar la aplicación de escritorio

```bash
python ship_detection_ui_v1.5.py
```

La aplicación permite:
- **Inferencia individual**: cargar una imagen y ver la predicción con nivel de confianza.
- **Evaluación por lote**: seleccionar una carpeta completa y obtener métricas agregadas (si las subcarpetas o nombres de archivo indican la etiqueta real).

## Notas

- El tamaño de imagen (80×80) es un requerimiento fijo del proyecto; las mejoras de accuracy se buscan por arquitectura y augmentación, no por resolución de entrada.
- La arquitectura del modelo en `ship_detection_ui_v1.5.py` debe coincidir exactamente con la definida en `train_ships_v1.5_finetune.py` para poder cargar los pesos guardados (`state_dict`).

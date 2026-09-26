import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# UI actualizada para consumir el modelo de v1.5 (fine-tuning end-to-end)
# en vez del pipeline sklearn de v1.3/v1.4. Ver notas al final del archivo.
# ==========================================================

IMG_SIZE = (80, 80)
MODEL_PATH = 'model/best_ship_finetuned.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Mismas 4 vistas de TTA usadas al medir el blind test en el entrenamiento,
# para que la accuracy que ves aqui sea comparable a la reportada al entrenar.
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


class ResNetFineTune(nn.Module):
    """
    Debe ser IDENTICA a la clase usada al entrenar (train_ships_v1.5_finetune.py):
    la arquitectura tiene que coincidir exactamente para poder cargar el
    state_dict guardado en MODEL_PATH.
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        base = models.resnet34(weights=None)  # los pesos reales vienen del state_dict
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


def load_model():
    """Carga la arquitectura y le inyecta los pesos entrenados mas recientes."""
    model = ResNetFineTune().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, pil_img, use_tta=True):
    """
    Reemplaza el viejo extract_features() + model.predict()/predict_proba()
    del pipeline sklearn: ahora el modelo predice directamente end-to-end.
    Con use_tta=True promedia probabilidades sobre 4 vistas (igual que en
    la evaluacion del blind test durante el entrenamiento).
    """
    view_transforms = tta_transforms if use_tta else [eval_transform]
    probs_sum = None
    with torch.no_grad():
        for t in view_transforms:
            tensor_img = t(pil_img).unsqueeze(0).to(device)
            logits = model(tensor_img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_sum = probs if probs_sum is None else probs_sum + probs
    probs_avg = probs_sum / len(view_transforms)
    pred = int(np.argmax(probs_avg))
    conf = float(probs_avg[pred])
    return pred, conf


# CustomTkinter UI Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Ship Detection AI - Local Desktop Engine (Offline)")
        self.geometry("1150x800")
        self.minsize(900, 650)

        # Cargar modelo fine-tuned mas reciente (.pt), ya NO el pipeline .pkl
        self.model = None
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model()
            except Exception as e:
                messagebox.showerror(
                    "Error de Carga",
                    f"No se pudo cargar el modelo ({MODEL_PATH}): {e}\n\n"
                    "Verifica que la clase ResNetFineTune de esta UI coincida "
                    "exactamente con la usada en el script de entrenamiento."
                )

        # Main Grid Layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=230, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Ship AI\nEngine v1.5",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 15))

        self.nav_btn_1 = ctk.CTkButton(
            self.sidebar_frame,
            text="Inferencia Individual",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=lambda: self.select_frame("inference")
        )
        self.nav_btn_1.grid(row=1, column=0, padx=20, pady=10)

        self.nav_btn_2 = ctk.CTkButton(
            self.sidebar_frame,
            text="Evaluación de Carpeta / Lote",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=lambda: self.select_frame("test_day")
        )
        self.nav_btn_2.grid(row=2, column=0, padx=20, pady=10)

        # Status Badge (ya no afirma un % fijo: se valida en cada evaluacion real)
        status_text = "Estado: Modelo Fine-Tuned Cargado" if self.model else "Estado: Modelo No Encontrado"
        status_color = "#10b981" if self.model else "#ef4444"
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=status_text,
            text_color=status_color,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_label.grid(row=5, column=0, padx=15, pady=20)

        # Main Content Frames
        self.inference_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.test_day_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")

        self.setup_inference_frame()
        self.setup_test_day_frame()

        self.select_frame("inference")

    def select_frame(self, name):
        if name == "inference":
            self.inference_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.test_day_frame.grid_forget()
            self.nav_btn_1.configure(fg_color=("gray75", "#2563eb"))
            self.nav_btn_2.configure(fg_color="transparent")
        else:
            self.test_day_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.inference_frame.grid_forget()
            self.nav_btn_2.configure(fg_color=("gray75", "#2563eb"))
            self.nav_btn_1.configure(fg_color="transparent")

    # ================== INFERENCIA INDIVIDUAL ==================
    def setup_inference_frame(self):
        self.inference_frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.inference_frame,
            text="Inferencia de Imagen Satelital",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        title.grid(row=0, column=0, pady=(0, 15))

        self.img_label = ctk.CTkLabel(
            self.inference_frame,
            text="Selecciona cualquier imagen satelital (.png, .jpg)",
            width=260,
            height=260,
            fg_color="#1e293b",
            corner_radius=12
        )
        self.img_label.grid(row=1, column=0, pady=10)

        btn_upload = ctk.CTkButton(
            self.inference_frame,
            text="📂 Cargar Imagen Satelital",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.upload_single_image
        )
        btn_upload.grid(row=2, column=0, pady=15)

        self.inf_result_label = ctk.CTkLabel(
            self.inference_frame,
            text="",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.inf_result_label.grid(row=3, column=0, pady=5)

        self.inf_conf_bar = ctk.CTkProgressBar(self.inference_frame, width=340, height=14)
        self.inf_conf_bar.grid(row=4, column=0, pady=10)
        self.inf_conf_bar.set(0)
        self.inf_conf_bar.grid_remove()

        self.inf_conf_text = ctk.CTkLabel(
            self.inference_frame,
            text="",
            font=ctk.CTkFont(size=14)
        )
        self.inf_conf_text.grid(row=5, column=0)

    def upload_single_image(self):
        if not self.model:
            messagebox.showerror("Error", f"¡El modelo no está cargado! Ejecuta el script de entrenamiento primero (debe generar {MODEL_PATH}).")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Archivos de Imagen", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        img = Image.open(file_path).convert('RGB')
        img_display = img.copy().resize((250, 250))
        ctk_img = ctk.CTkImage(light_image=img_display, dark_image=img_display, size=(250, 250))

        self.img_label.configure(image=ctk_img, text="")

        # Predicción end-to-end (con TTA) usando el modelo fine-tuned
        start = time.time()
        pred, conf = predict_image(self.model, img, use_tta=True)
        elapsed = (time.time() - start) * 1000

        if pred == 1:
            self.inf_result_label.configure(text="🚢 BARCO DETECTADO (SHIP)", text_color="#10b981")
            self.inf_conf_bar.configure(progress_color="#10b981")
        else:
            self.inf_result_label.configure(text="🌊 SIN BARCO (NO SHIP)", text_color="#ef4444")
            self.inf_conf_bar.configure(progress_color="#ef4444")

        self.inf_conf_bar.grid()
        self.inf_conf_bar.set(conf)
        self.inf_conf_text.configure(
            text=f"Nivel de Confianza: {conf*100:.2f}%\nTiempo de inferencia local (con TTA): {elapsed:.1f} ms"
        )

    # ================== EVALUACIÓN DÍA DE PRUEBA / CARPETA ==================
    def setup_test_day_frame(self):
        self.test_day_frame.grid_columnconfigure(0, weight=1)
        self.test_day_frame.grid_rowconfigure(3, weight=1)

        title = ctk.CTkLabel(
            self.test_day_frame,
            text="Procesamiento de Carpeta / Lote",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        title.grid(row=0, column=0, pady=(0, 5))

        info = ctk.CTkLabel(
            self.test_day_frame,
            text="Selecciona cualquier carpeta con imágenes (con o sin subcarpetas, nombres personalizados o flat).\nEl sistema clasificará cada imagen y calculará métricas si las etiquetas reales están presentes."
        )
        info.grid(row=1, column=0, pady=5)

        btn_folder = ctk.CTkButton(
            self.test_day_frame,
            text="📁 Seleccionar Carpeta de Imágenes",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.load_test_directory
        )
        btn_folder.grid(row=2, column=0, pady=10)

        self.results_scroll = ctk.CTkScrollableFrame(self.test_day_frame, fg_color="transparent")
        self.results_scroll.grid(row=3, column=0, sticky="nsew", pady=10)
        self.results_scroll.grid_columnconfigure((0, 1, 2), weight=1)

    def load_test_directory(self):
        if not self.model:
            messagebox.showerror("Error", f"¡El modelo no está cargado! Ejecuta el script de entrenamiento primero (debe generar {MODEL_PATH}).")
            return

        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        lbl_status = ctk.CTkLabel(
            self.results_scroll,
            text="Analizando imágenes localmente con la IA... Por favor espera.",
            font=ctk.CTkFont(size=16)
        )
        lbl_status.grid(row=0, column=0, columnspan=3, pady=20)
        self.update()

        true_labels = []
        predictions = []
        has_true_labels = True
        results_detail = []

        start_time = time.time()
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    parent_folder = os.path.basename(root).lower()
                    fname = file.lower()

                    # Intentar determinar la etiqueta real verdadera
                    t_label = None
                    if 'no_ship' in parent_folder or 'noship' in parent_folder:
                        t_label = 0
                    elif 'ship' in parent_folder:
                        t_label = 1
                    elif fname.startswith('1__') or fname.startswith('ship_'):
                        t_label = 1
                    elif fname.startswith('0__') or fname.startswith('noship_') or fname.startswith('no_ship_'):
                        t_label = 0
                    else:
                        has_true_labels = False

                    if t_label is not None:
                        true_labels.append(t_label)

                    img = Image.open(path).convert('RGB')
                    pred, conf = predict_image(self.model, img, use_tta=True)

                    predictions.append(pred)
                    results_detail.append({
                        "filename": file,
                        "pred": pred,
                        "conf": conf,
                        "true_label": t_label
                    })

        total_time = time.time() - start_time
        lbl_status.destroy()

        if len(predictions) == 0:
            ctk.CTkLabel(self.results_scroll, text="¡No se encontraron imágenes en la carpeta!").grid(row=0, column=0)
            return

        ships_count = sum(1 for p in predictions if p == 1)
        noship_count = sum(1 for p in predictions if p == 0)

        # Caso A: Se tienen etiquetas reales conocidas (Subcarpetas ship/no_ship o prefijos 1__/0__)
        if has_true_labels and len(true_labels) == len(predictions):
            acc = accuracy_score(true_labels, predictions)
            prec = precision_score(true_labels, predictions, zero_division=0)
            rec = recall_score(true_labels, predictions, zero_division=0)
            cm = confusion_matrix(true_labels, predictions)

            metrics_frame = ctk.CTkFrame(self.results_scroll, fg_color="#1e293b", corner_radius=12)
            metrics_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=10, padx=10)
            metrics_frame.grid_columnconfigure((0, 1, 2), weight=1)

            ctk.CTkLabel(
                metrics_frame,
                text=f"Exactitud (Accuracy)\n{acc*100:.2f}%",
                font=("Arial", 22, "bold"),
                text_color="#60a5fa"
            ).grid(row=0, column=0, pady=20)

            ctk.CTkLabel(
                metrics_frame,
                text=f"Precisión (Precision)\n{prec*100:.2f}%",
                font=("Arial", 22, "bold"),
                text_color="#34d399"
            ).grid(row=0, column=1, pady=20)

            ctk.CTkLabel(
                metrics_frame,
                text=f"Sensibilidad (Recall)\n{rec*100:.2f}%",
                font=("Arial", 22, "bold"),
                text_color="#f472b6"
            ).grid(row=0, column=2, pady=20)

            info = (
                f"Imágenes analizadas: {len(predictions)} | Barcos detectados: {ships_count} | Sin barco: {noship_count}\n"
                f"Tiempo total: {total_time:.2f}s ({total_time/len(predictions)*1000:.1f} ms/img, incluye TTA)\n\n"
                f"Matriz de Confusión:\n"
                f"TN (Verdaderos Negativos): {cm[0][0]}   |   FP (Falsos Positivos): {cm[0][1] if cm.shape[1]>1 else 0}\n"
                f"FN (Falsos Negativos): {cm[1][0] if cm.shape[0]>1 else 0}   |   TP (Verdaderos Positivos): {cm[1][1] if cm.shape[0]>1 and cm.shape[1]>1 else 0}"
            )
            ctk.CTkLabel(self.results_scroll, text=info, font=("Consolas", 14), justify="center").grid(row=1, column=0, columnspan=3, pady=15)

            if acc >= 0.98:
                ctk.CTkLabel(
                    self.results_scroll,
                    text="🏆 OBJETIVO CUMPLIDO: > 98% DE ACCURACY ALCANZADO 🏆",
                    font=("Arial", 20, "bold"),
                    text_color="#10b981"
                ).grid(row=2, column=0, columnspan=3, pady=15)

        # Caso B: Carpeta plana sin etiquetas verdaderas (Nombres de archivo personalizados sin prefijo)
        else:
            summary_frame = ctk.CTkFrame(self.results_scroll, fg_color="#1e293b", corner_radius=12)
            summary_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=10, padx=10)
            summary_frame.grid_columnconfigure((0, 1, 2), weight=1)

            ctk.CTkLabel(
                summary_frame,
                text=f"Total Procesadas\n{len(predictions)} Img",
                font=("Arial", 22, "bold"),
                text_color="#60a5fa"
            ).grid(row=0, column=0, pady=20)

            ctk.CTkLabel(
                summary_frame,
                text=f"Barcos Detectados 🚢\n{ships_count} ({ships_count/len(predictions)*100:.1f}%)",
                font=("Arial", 22, "bold"),
                text_color="#10b981"
            ).grid(row=0, column=1, pady=20)

            ctk.CTkLabel(
                summary_frame,
                text=f"Sin Barco 🌊\n{noship_count} ({noship_count/len(predictions)*100:.1f}%)",
                font=("Arial", 22, "bold"),
                text_color="#ef4444"
            ).grid(row=0, column=2, pady=20)

            info = f"Tiempo de análisis en lote: {total_time:.2f}s ({total_time/len(predictions)*1000:.1f} ms por imagen, incluye TTA)"
            ctk.CTkLabel(self.results_scroll, text=info, font=("Consolas", 14)).grid(row=1, column=0, columnspan=3, pady=10)

            # Tabla de resultados detalle
            lbl_detail = ctk.CTkLabel(
                self.results_scroll,
                text="Resultados de Predicción por Imagen:",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            lbl_detail.grid(row=2, column=0, columnspan=3, pady=(15, 5))

            table_frame = ctk.CTkFrame(self.results_scroll, fg_color="#0f172a")
            table_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
            table_frame.grid_columnconfigure((0, 1, 2), weight=1)

            # Encabezados
            ctk.CTkLabel(table_frame, text="Nombre de Archivo", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=5)
            ctk.CTkLabel(table_frame, text="Predicción IA", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=10, pady=5)
            ctk.CTkLabel(table_frame, text="Confianza", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=10, pady=5)

            # Mostrar hasta 50 filas
            for i, res in enumerate(results_detail[:50]):
                p_text = "🚢 Barco" if res["pred"] == 1 else "🌊 No Barco"
                p_color = "#10b981" if res["pred"] == 1 else "#ef4444"

                ctk.CTkLabel(table_frame, text=res["filename"], font=("Consolas", 11)).grid(row=i+1, column=0, padx=5, pady=2)
                ctk.CTkLabel(table_frame, text=p_text, font=("Arial", 11, "bold"), text_color=p_color).grid(row=i+1, column=1, padx=5, pady=2)
                ctk.CTkLabel(table_frame, text=f"{res['conf']*100:.1f}%", font=("Consolas", 11)).grid(row=i+1, column=2, padx=5, pady=2)

            if len(results_detail) > 50:
                ctk.CTkLabel(
                    table_frame,
                    text=f"... y {len(results_detail)-50} imágenes más procesadas exitosamente.",
                    font=("Arial", 11, "italic")
                ).grid(row=52, column=0, columnspan=3, pady=5)


if __name__ == "__main__":
    app = App()
    app.mainloop()

# ==========================================================
# QUE SE CORRIGIO Y POR QUE NO SE ACTUALIZABA LA INFORMACION
# ==========================================================
# La causa raiz: esta UI y el script de entrenamiento habian quedado
# desincronizados. La UI seguia asumiendo el flujo viejo (v1.3/v1.4):
#   - Cargaba 'model/best_ship_ml.pkl' con joblib (un Pipeline de sklearn).
#   - Extraia features con un ResNet18 CONGELADO (sin maxpool eliminado,
#     sin fine-tuning) y llamaba a model.predict()/model.predict_proba().
#
# Pero el entrenamiento mas reciente (v1.5) ya no genera ese .pkl: genera
# 'model/best_ship_finetuned.pt', un state_dict de PyTorch de un modelo
# ResNetFineTune (layer3/layer4 entrenados end-to-end + cabeza propia). Por
# eso, aunque reentrenaras el modelo una y otra vez, la UI seguia mostrando
# resultados del modelo viejo (o fallaba al cargar): estaba apuntando a un
# archivo/formato que el entrenamiento actual ya no produce.
#
# Cambios en esta version:
# 1) MODEL_PATH apunta a 'model/best_ship_finetuned.pt' (el que realmente
#    genera el entrenamiento actual), no al .pkl viejo.
# 2) Se agrego la clase ResNetFineTune IDENTICA a la del script de
#    entrenamiento. Esto es obligatorio: un state_dict de PyTorch solo se
#    puede cargar en una arquitectura con la misma estructura de capas.
#    Si vuelves a modificar la arquitectura en el script de entrenamiento,
#    tienes que replicar el cambio aqui tambien, o el load_state_dict fallara.
# 3) Se elimino por completo el flujo extract_features() + predict()/
#    predict_proba() de sklearn. Ahora predict_image() hace un forward pass
#    directo por el modelo (end-to-end) y usa softmax para obtener
#    probabilidades, con TTA opcional (4 vistas) para que la accuracy que
#    ves en la carpeta de evaluacion sea comparable a la reportada durante
#    el entrenamiento (que tambien usa TTA en el blind test).
# 4) El badge de estado ya no afirma "98%+ Acc" de forma fija (ese numero
#    quedaba desactualizado apenas cambiabas el modelo); ahora solo indica
#    si el modelo cargo correctamente, y la accuracy real se calcula en
#    vivo cuando evaluas una carpeta con etiquetas conocidas.
#
# RECOMENDACION A FUTURO
# ==========================================================
# Para evitar que esto vuelva a pasar, seria buena idea que el script de
# entrenamiento guarde tambien un pequeno JSON junto al .pt (por ejemplo
# model/model_info.json con la version de arquitectura, fecha de
# entrenamiento y accuracy en blind test), y que esta UI lo lea al arrancar
# para mostrar esa informacion real en vez de un texto fijo en el codigo.
# ==========================================================
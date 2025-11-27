# gui_prediccion_tk.py
"""
Interfaz Tkinter para predecir platos con el modelo PyTorch.
- Carga modelo 'modelo_completo.pth' (debe existir en la carpeta del proyecto)
- Intenta leer los CSVs de ingredientes y nutrición en varias rutas comunes
- Muestra Top-3 predicciones y la info nutricional/ingredientes
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import numpy as np

# Intento de importar el modelo (usa el ComidaClassifier corregido)
# Asegúrate que entrenar_modelo_pytorch.py esté en la misma carpeta o en PYTHONPATH
# Definir la clase modelo directamente para asegurar compatibilidad
class ComidaClassifier(nn.Module):
    """
    Modelo de clasificación de comidas usando transfer learning con ResNet50
    Compatible con modelo_completo.pth
    """
    def __init__(self, num_classes):
        super(ComidaClassifier, self).__init__()
        from torchvision import models
        self.resnet = models.resnet50(pretrained=False)

        # Congelar capas iniciales (como se hace en el entrenamiento original)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Reemplazar la capa final con la arquitectura compatible
        # Basado en análisis del modelo guardado: tiene BatchNorm y diferentes dimensiones
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),              # 0
            nn.Linear(num_features, 512), # 1
            nn.ReLU(),                    # 2
            nn.BatchNorm1d(512),          # 3
            nn.Dropout(0.3),              # 4
            nn.Linear(512, 256),          # 5
            nn.ReLU(),                    # 6
            nn.Linear(256, num_classes)   # 7
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

# ComidaClassifier ya está definido en este archivo
ComidaClassifier = ComidaClassifier

# ------------------ Config ------------------
MODEL_PATHS = [
    "modelo_completo.pth",
    "mejor_modelo.pth",
    os.path.join("model", "modelo_completo.pth"),
    os.path.join("models", "modelo_completo.pth")
]

# posibles ubicaciones de CSVs (intenta varias)
POSSIBLE_ING_PATHS = [
    "dataset_ingredientes.csv",
    os.path.join("dataset", "dataset_ingredientes.csv"),
    os.path.join("dataset", "dataset_ingredientes.csv"),
    os.path.join("data", "dataset_ingredientes.csv")
]
POSSIBLE_NUT_PATHS = [
    "nutricion/datos_nutricionales.csv",
    os.path.join("dataset", "datos_nutricionales.csv"),
    os.path.join("dataset", "datos_nutricionales.csv"),
    "datos_nutricionales.csv",
    os.path.join("data", "datos_nutricionales.csv")
]

IMAGE_SIZE = 224

# ------------------ Helpers ------------------
def try_load_csv(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception as e:
                print(f"Error leyendo {p}: {e}")
    return None, None

def load_model(model_paths=MODEL_PATHS, device="cpu"):
    """Carga el checkpoint y devuelve (model, classes, mapeo)"""
    path = None
    for p in model_paths:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"No se encontró checkpoint en ninguna de las rutas: {model_paths}")

    checkpoint = torch.load(path, map_location=device)
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
    else:
        # intentar inferir
        classes = checkpoint.get('class_indices', None)
        if classes is None:
            raise KeyError("El checkpoint no contiene 'classes' ni 'class_indices'")

    mapeo = checkpoint.get('mapeo', None)

    if ComidaClassifier is None:
        raise ImportError("ComidaClassifier no está disponible. Asegúrate que 'entrenar_modelo_pytorch.py' esté presente y sin errores.")

    model = ComidaClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes, mapeo, path

def predict_topk(model, img_path, classes, k=3, device="cpu"):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    t = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    topk_idx = probs.argsort()[::-1][:k]
    return [(classes[i], float(probs[i])) for i in topk_idx]

def find_ingredient_row(ingred_df, nombre_plato, mapeo=None):
    if ingred_df is None:
        return None
    # Primero intentar coincidencia exacta en 'plato' columna
    col = ingred_df.columns
    if 'plato' in col:
        rows = ingred_df[ingred_df['plato'] == nombre_plato]
        if not rows.empty:
            return rows.iloc[0]
    # Si hay mapping de clase a nombre original (mapeo), usarlo
    if mapeo:
        # mapeo puede ser dict csv_name -> carpeta, invertimos
        inv = {v: k for k, v in mapeo.items()}
        original = inv.get(nombre_plato, None)
        if original:
            rows = ingred_df[ingred_df['plato'] == original]
            if not rows.empty:
                return rows.iloc[0]
    # intentar coincidencia normalizada
    def norm(s):
        return str(s).strip().lower().replace("_", " ").replace("-", " ")
    target = norm(nombre_plato)
    for _, r in ingred_df.iterrows():
        if norm(r.get('plato', '')) == target:
            return r
    return None

def parse_ingredientes(cell):
    # intenta interpretar listas guardadas como strings
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return list(cell)
    s = str(cell)
    # si es formato "['a','b']" intentar eval (con seguridad)
    try:
        val = eval(s)
        if isinstance(val, (list, tuple)):
            return list(val)
    except Exception:
        pass
    # fallback: separar por comas
    return [x.strip() for x in s.split(",") if x.strip()]

# ------------------ GUI ------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Platos - Interfaz")
        self.root.geometry("900x600")

        # Cargar recursos (modelo y CSVs) en background opcional
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.classes = None
        self.mapeo = None
        self.model_path_used = None

        # Panel izquierdo: imagen + botones
        left = ttk.Frame(root, padding=8)
        left.pack(side="left", fill="both", expand=False)

        self.img_label = ttk.Label(left, text="No hay imagen", width=60)
        self.img_label.pack(padx=5, pady=5)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=6)
        ttk.Button(btn_frame, text="Abrir imagen", command=self.on_open).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Predecir (Top-3)", command=self.on_predict).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cargar modelo", command=self.on_load_model).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Modelo: no cargado")
        ttk.Label(left, textvariable=self.status_var).pack(pady=6)

        # Panel derecho: resultados
        right = ttk.Frame(root, padding=8)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(right, text="Top 3 predicciones:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.topk_text = tk.Text(right, height=5, wrap="word")
        self.topk_text.pack(fill="x", pady=4)

        ttk.Label(right, text="Ingredientes:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(8,0))
        self.ing_text = tk.Text(right, height=8, wrap="word")
        self.ing_text.pack(fill="both", expand=False, pady=4)

        ttk.Label(right, text="Información nutricional:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(8,0))
        self.nut_text = tk.Text(right, height=6, wrap="word")
        self.nut_text.pack(fill="both", expand=False, pady=4)

        # Intenta cargar CSVs automáticamente
        self.ingred_df, ing_path = try_load_csv(POSSIBLE_ING_PATHS)
        self.nut_df, nut_path = try_load_csv(POSSIBLE_NUT_PATHS)
        if self.ingred_df is not None:
            print("Ingredientes cargados desde:", ing_path)
        if self.nut_df is not None:
            print("Nutrición cargada desde:", nut_path)

        # Guardar ruta imagen seleccionada
        self.current_image_path = None

    def on_open(self):
        path = filedialog.askopenfilename(title="Selecciona imagen",
                                          filetypes=[("Imagen", "*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")])
        if not path:
            return
        self.current_image_path = path
        # mostrar imagen en label
        img = Image.open(path).convert("RGB")
        # redimensionar para preview manteniendo aspect
        w, h = img.size
        max_w = 400
        max_h = 400
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_resized)
        self.img_label.configure(image=self.photo, text="")
        self.topk_text.delete("1.0", tk.END)
        self.ing_text.delete("1.0", tk.END)
        self.nut_text.delete("1.0", tk.END)
        self.status_var.set(f"Imagen: {os.path.basename(path)} — modelo: {'cargado' if self.model else 'no cargado'}")

    def on_load_model(self):
        try:
            model, classes, mapeo, used = load_model(MODEL_PATHS, device=self.device)
            self.model = model
            self.classes = classes
            self.mapeo = mapeo
            self.model_path_used = used
            self.status_var.set(f"Modelo cargado: {os.path.basename(used)} — {len(classes)} clases")
            messagebox.showinfo("Modelo cargado", f"Modelo cargado desde: {used}\nClases: {len(classes)}")
        except Exception as e:
            messagebox.showerror("Error cargando modelo", str(e))

    def on_predict(self):
        if self.current_image_path is None:
            messagebox.showwarning("Sin imagen", "Primero selecciona una imagen (Abrir imagen).")
            return
        if self.model is None:
            # intentar cargar automáticamente
            try:
                self.on_load_model()
            except Exception:
                messagebox.showwarning("Modelo no cargado", "No hay modelo cargado y no se pudo cargar automáticamente.")
                return
        try:
            top3 = predict_topk(self.model, self.current_image_path, self.classes, k=3, device=self.device)
            self.topk_text.delete("1.0", tk.END)
            for i, (name, prob) in enumerate(top3, 1):
                self.topk_text.insert(tk.END, f"{i}. {name} — {prob:.2%}\n")

            # obtener ingredientes y nutricion para la clase candidata principal
            main_class = top3[0][0]
            # main_class es el nombre de carpeta/clase. Buscar fila en CSVs
            ing_row = find_ingredient_row(self.ingred_df, main_class, mapeo=self.mapeo)
            self.ing_text.delete("1.0", tk.END)
            if ing_row is not None:
                if 'ingredientes' in ing_row.index:
                    ings = parse_ingredientes(ing_row['ingredientes'])
                    if ings:
                        for it in ings:
                            self.ing_text.insert(tk.END, f" - {it}\n")
                    else:
                        self.ing_text.insert(tk.END, "No hay ingredientes listados\n")
                else:
                    # imprimir todo el row si no hay columna 'ingredientes'
                    self.ing_text.insert(tk.END, str(ing_row.to_dict()))
            else:
                self.ing_text.insert(tk.END, "No se encontró información de ingredientes\n")

            # nutrición
            nut_row = None
            if self.nut_df is not None:
                # buscar por 'plato' semejante
                nut_row = find_ingredient_row(self.nut_df, main_class, mapeo=self.mapeo)
            self.nut_text.delete("1.0", tk.END)
            if nut_row is not None:
                # intentar columnas comunes
                for col in ['calorias', 'proteina', 'carbohidratos', 'grasa']:
                    if col in nut_row.index:
                        self.nut_text.insert(tk.END, f"{col.capitalize()}: {nut_row[col]}\n")
                if self.nut_text.get("1.0", tk.END).strip() == "":
                    # imprimir todo el row
                    self.nut_text.insert(tk.END, str(nut_row.to_dict()))
            else:
                self.nut_text.insert(tk.END, "No se encontró información nutricional\n")

        except Exception as e:
            messagebox.showerror("Error en predicción", str(e))


# ------------------ Run ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

"""
Script para predecir platos usando el modelo entrenado con PyTorch y conectar con los datasets de ingredientes y nutrición
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Cargar los datasets
ingredientes_df = pd.read_csv('dataset_ingredientes.csv')
nutricion_df = pd.read_csv('nutricion/datos_nutricionales.csv')

# Modelo de clasificación (mismo que en el entrenamiento)
class ComidaClassifier(nn.Module):
    """
    Modelo de clasificación de comidas usando transfer learning con ResNet50
    """
    def __init__(self, num_classes):
        super(ComidaClassifier, self).__init__()
        from torchvision import models
        self.resnet = models.resnet50(pretrained=False)
        
        # Reemplazar la capa final
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

def crear_mapeo_nombres():
    """
    Crea un mapeo entre los nombres en los CSV y las carpetas de imágenes
    """
    nombre_mapeo = {
        'Aji de Pataskha': 'Aji_de_pataskha',
        'Caldo de Bagre': 'caldo_de_bagre',
        'Cazuela': 'Cazuelas',
        'Chairo': 'Chairo',
        'Charquekan Orureño': 'Charquekan_Orureno',
        'Chicharrón': 'chicharron',
        'Chorizos': 'Chorizos',
        'Cunape': 'cunape',
        'Empanada de Arroz': 'empanada_de_arroz',
        'Falso Conejo': 'Falso_Conejo',
        'Fricase': 'fricase',
        'Fritanga': 'fritanga',
        'Karapecho': 'Karapecho',
        'Keperi Beniano': 'keperi_beniano',
        'La Kalapurka': 'La_kalapurka',
        'Locro': 'locro',
        'Majao': 'majao',
        'Mondongo Chuquisaqueño': 'Mondongo',
        'Pacu Frito': 'pacu',
        'Pejerrey': 'Pejerrey',
        'Picana': 'Picana',
        'Picante de Pollo': 'Picante_de_Pollo',
        'Pique Macho': 'pique',
        'Ranga': 'Ranga',
        'Saice': 'Saice',
        'Silpancho': 'silpancho',
        'Sopa de Maní': 'Sopa_de_Mani',
        'Sopa de Quinua': 'Sopa_De_Quinua'
    }
    return nombre_mapeo

def cargar_modelo():
    """
    Carga el modelo entrenado
    """
    try:
        checkpoint = torch.load('modelo_completo.pth', map_location=torch.device('cpu'))
        classes = checkpoint['classes']
        model = ComidaClassifier(num_classes=len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, classes
    except FileNotFoundError:
        print("Error: No se encontró el modelo entrenado 'modelo_completo.pth'")
        print("Debes entrenar el modelo primero usando 'entrenar_modelo_pytorch.py'")
        return None, None

def predecir_plato(modelo, img_path, classes):
    """
    Predice el plato en una imagen
    """
    # Transformaciones para la predicción
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar y preprocesar la imagen
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Añadir dimensión de batch
    
    # Hacer predicción
    with torch.no_grad():
        outputs = modelo(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        predicted_class = classes[predicted_class_idx]
    
    return predicted_class, confidence

def obtener_info_plato(nombre_plato):
    """
    Obtiene información de ingredientes y nutrición para un plato
    """
    # Buscar ingredientes
    ingredientes_row = ingredientes_df[ingredientes_df['plato'] == nombre_plato]
    ingredientes = []
    if not ingredientes_row.empty:
        ingredientes_str = ingredientes_row['ingredientes'].values[0]
        # Convertir string de lista a lista real
        try:
            ingredientes = eval(ingredientes_str)
        except:
            ingredientes = []

    # Buscar información nutricional
    nutricion_row = nutricion_df[nutricion_df['plato'] == nombre_plato]
    nutricion = {}
    if not nutricion_row.empty:
        nutricion = {
            'calorias': int(nutricion_row['calorias'].values[0]),
            'proteina': int(nutricion_row['proteina'].values[0]),
            'carbohidratos': int(nutricion_row['carbohidratos'].values[0]),
            'grasa': int(nutricion_row['grasa'].values[0])
        }
    
    return ingredientes, nutricion

def procesar_imagen(modelo, img_path, classes):
    """
    Procesa una imagen completa: predicción + información nutricional
    """
    # Predecir el plato
    predicted_class, confidence = predecir_plato(modelo, img_path, classes)
    
    # Buscar el nombre original en el CSV (deshaciendo la normalización si es necesario)
    nombre_mapeo = crear_mapeo_nombres()
    nombre_original = None
    for original, normalizado in nombre_mapeo.items():
        if normalizado == predicted_class:
            nombre_original = original
            break
    
    if nombre_original is None:
        # Si no encontramos un mapeo exacto, asumimos que el nombre de clase es el nombre original
        nombre_original = predicted_class
    
    # Obtener información del plato
    ingredientes, nutricion = obtener_info_plato(nombre_original)
    
    return {
        'nombre_plato': nombre_original,
        'clase_predicha': predicted_class,
        'confianza': confidence,
        'ingredientes': ingredientes,
        'nutricion': nutricion
    }

def mostrar_resultados(resultado):
    """
    Muestra los resultados de la predicción
    """
    print("="*60)
    print("RESULTADOS DE LA PREDICCIÓN")
    print("="*60)
    print(f"Plato identificado: {resultado['nombre_plato']}")
    print(f"Confianza: {resultado['confianza']:.2%}")
    print()
    
    print("INGREDIENTES:")
    if resultado['ingredientes']:
        for ingrediente in resultado['ingredientes']:
            print(f"  - {ingrediente}")
    else:
        print("  No disponible")
    print()
    
    print("INFORMACIÓN NUTRICIONAL:")
    if resultado['nutricion']:
        print(f"  Calorías: {resultado['nutricion']['calorias']} kcal")
        print(f"  Proteína: {resultado['nutricion']['proteina']} g")
        print(f"  Carbohidratos: {resultado['nutricion']['carbohidratos']} g")
        print(f"  Grasa: {resultado['nutricion']['grasa']} g")
    else:
        print("  No disponible")
    print("="*60)

def main():
    """
    Función principal para predecir un plato a partir de una imagen
    """
    # Cargar modelo entrenado
    modelo, classes = cargar_modelo()
    if modelo is None:
        return
    
    print("Modelo cargado exitosamente")
    print(f"Número de clases: {len(classes)}")
    print()
    
    print("Ejemplo de uso:")
    print("Ingresa la ruta a una imagen para clasificarla:")
    
    # Ruta de ejemplo - puedes cambiarla por cualquier imagen de tu dataset
    img_path = input("Ruta de la imagen: ").strip()
    
    if not os.path.exists(img_path):
        print("La ruta especificada no existe.")
        return
    
    # Procesar la imagen
    try:
        resultado = procesar_imagen(modelo, img_path, classes)
        mostrar_resultados(resultado)
        
        # Mostrar la imagen
        img = Image.open(img_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Imagen clasificada como: {resultado['nombre_plato']}\nConfianza: {resultado['confianza']:.2%}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    main()
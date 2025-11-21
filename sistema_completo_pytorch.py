"""
Sistema completo para reconocimiento de comidas típicas bolivianas con PyTorch
- Entrenamiento de modelo de visión por computadora
- Clasificación de imágenes de platos
- Obtención de ingredientes y valores nutricionales
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import random

# Cargar los datasets
ingredientes_df = pd.read_csv('dataset_ingredientes.csv')
nutricion_df = pd.read_csv('nutricion/datos_nutricionales.csv')

# Configuración
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Fijar semillas para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

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

class ComidasBolivianasDataset(Dataset):
    """
    Dataset personalizado para comidas típicas bolivianas
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class ComidaClassifier(nn.Module):
    """
    Modelo de clasificación de comidas usando transfer learning con ResNet50
    """
    def __init__(self, num_classes):
        super(ComidaClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        # Congelar capas iniciales
        for param in self.resnet.parameters():
            param.requires_grad = False
            
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

def preparar_datos():
    """
    Prepara los datos para entrenamiento, resolviendo discrepancias de nombres
    """
    data_dir = 'dataset_imagenes'
    
    # Obtener carpetas reales
    carpetas_reales = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Mapear nombres de platos a carpetas reales
    nombre_mapeo = crear_mapeo_nombres()
    mapeo_final = {}
    
    for plato_csv in ingredientes_df['plato']:
        carpeta_normalizada = nombre_mapeo.get(plato_csv, plato_csv.lower().replace(' ', '_').replace('-', '_'))
        if carpeta_normalizada in carpetas_reales:
            mapeo_final[plato_csv] = carpeta_normalizada
    
    print("Mapeo de platos a carpetas:")
    for plato, carpeta in mapeo_final.items():
        print(f"  {plato} -> {carpeta}")
    
    # Crear lista de imágenes y etiquetas
    image_paths = []
    labels = []
    class_names = []
    
    for idx, (plato, carpeta) in enumerate(mapeo_final.items()):
        class_names.append(carpeta)
        class_path = os.path.join(data_dir, carpeta)
        if os.path.exists(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.JPG')):
                    img_path = os.path.join(class_path, img_name)
                    image_paths.append(img_path)
                    labels.append(idx)
    
    print(f"\nTotal de imágenes encontradas: {len(image_paths)}")
    print(f"Número de clases: {len(class_names)}")
    
    return image_paths, labels, class_names, mapeo_final

def entrenar_modelo():
    """
    Entrena el modelo de clasificación de imágenes
    """
    print("Preparando datos...")
    image_paths, labels, classes, mapeo = preparar_datos()
    
    if len(image_paths) == 0:
        print("No se encontraron imágenes válidas para entrenamiento")
        return None, None, None, None
    
    # Transformaciones para entrenamiento
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transformaciones para validación
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dividir datos en entrenamiento y validación
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    
    # Crear datasets
    train_dataset = ComidasBolivianasDataset(train_paths, train_labels, train_transform)
    val_dataset = ComidasBolivianasDataset(val_paths, val_labels, val_transform)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Crear modelo
    print("Creando modelo...")
    model = ComidaClassifier(num_classes=len(classes))
    
    # Definir dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model = model.to(device)
    
    # Definir criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler para reducir learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-7)
    
    # Variables para seguimiento del mejor modelo
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("Iniciando entrenamiento...")
    for epoch in range(EPOCHS):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Época {epoch+1}/{EPOCHS} - Entrenamiento')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Época {epoch+1}/{EPOCHS} - Validación')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': classes,
                'mapeo': mapeo
            }, 'mejor_modelo.pth')
            print(f"Nuevo mejor modelo guardado con precisión: {val_acc:.2f}%")
        
        # Registrar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Época {epoch+1}/{EPOCHS}:')
        print(f'  Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print()
    
    # Cargar el mejor modelo
    checkpoint = torch.load('mejor_modelo.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Guardar modelo completo
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'mapeo': mapeo,
        'class_indices': {name: idx for idx, name in enumerate(classes)}
    }, 'modelo_completo.pth')
    
    # Guardar historial de entrenamiento
    historial = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    with open('historial_entrenamiento.pkl', 'wb') as f:
        pickle.dump(historial, f)
    
    print(f"Entrenamiento completado! Mejor precisión: {best_val_acc:.2f}%")
    
    # Graficar resultados
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Precisión Entrenamiento')
    plt.plot(val_accuracies, label='Precisión Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Pérdida Entrenamiento')
    plt.plot(val_losses, label='Pérdida Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('graficas_entrenamiento.png')
    plt.show()
    
    return model, historial, classes, mapeo

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    modelo.eval()
    
    with torch.no_grad():
        outputs = modelo(img_tensor.to(device))
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

def procesar_imagen_completo(modelo, img_path, classes, mapeo):
    """
    Proceso completo: predicción + información nutricional
    """
    # Predecir plato
    clase_predicha, confianza = predecir_plato(modelo, img_path, classes)
    
    # Buscar nombre original en el mapeo
    nombre_original = None
    for csv_name, dir_name in mapeo.items():
        if dir_name == clase_predicha:
            nombre_original = csv_name
            break
    
    if nombre_original is None:
        nombre_original = clase_predicha
    
    # Obtener información del plato
    ingredientes, nutricion = obtener_info_plato(nombre_original)
    
    return {
        'nombre_plato': nombre_original,
        'clase_predicha': clase_predicha,
        'confianza': confianza,
        'ingredientes': ingredientes,
        'nutricion': nutricion
    }

def mostrar_resultados(resultado):
    """
    Muestra los resultados de forma clara
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
    Función principal del sistema
    """
    print("Sistema de Reconocimiento de Comidas Típicas Bolivianas (PyTorch)")
    print("1. Entrenar modelo")
    print("2. Clasificar imagen")
    
    opcion = input("\nSelecciona una opción (1 o 2): ").strip()
    
    if opcion == '1':
        print("\nIniciando entrenamiento del modelo...")
        modelo, historial, classes, mapeo = entrenar_modelo()
        if modelo is not None:
            print("Modelo entrenado y guardado exitosamente!")
        else:
            print("Error en el entrenamiento.")
    
    elif opcion == '2':
        try:
            # Cargar modelo
            checkpoint = torch.load('modelo_completo.pth', map_location=torch.device('cpu'))
            classes = checkpoint['classes']
            mapeo = checkpoint['mapeo']
            model = ComidaClassifier(num_classes=len(classes))
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Modelo cargado exitosamente")
            
            # Pedir ruta de imagen
            img_path = input("Ingresa la ruta de la imagen a clasificar: ").strip()
            
            if not os.path.exists(img_path):
                print("La ruta especificada no existe.")
                return
            
            # Procesar imagen
            resultado = procesar_imagen_completo(model, img_path, classes, mapeo)
            mostrar_resultados(resultado)
            
            # Mostrar imagen
            img = Image.open(img_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"{resultado['nombre_plato']}\nConfianza: {resultado['confianza']:.2%}")
            plt.axis('off')
            plt.show()
        
        except FileNotFoundError:
            print("Error: No se encontró un modelo entrenado. Debes entrenar el modelo primero (opción 1).")
        except Exception as e:
            print(f"Error al cargar el modelo o procesar la imagen: {str(e)}")
    
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
"""
Script para entrenar un modelo de visión por computadora usando PyTorch que clasifique comidas típicas bolivianas
"""
import os
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
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import random

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
        'Cazuela': 'Cazuelas',  # Ajuste basado en la discrepancia encontrada
        'Chairo': 'Chairo',
        'Charquekan Orureño': 'Charquekan_Orureno',
        'Chicharrón': 'chicharron',  # Corregido: "Chicharrn" en CSV a "chicharron" en carpeta
        'Chorizos': 'Chorizos',
        'Cunape': 'cunape',
        'Empanada de Arroz': 'empanada_de_arroz',
        'Falso Conejo': 'Falso_Conejo',
        'Fricase': 'fricase',
        'Fritanga': 'fritanga',
        'Karapecho': 'Karapecho',
        'Keperi Beniano': 'keperi_beniano',
        'La Kalapurka': 'La_kalapurka',  # Ajustado: "La Kalapurka" en CSV a "La_kalapurka" en carpeta
        'Locro': 'locro',
        'Majao': 'majao',
        'Mondongo Chuquisaqueño': 'Mondongo',  # Ajuste basado en la discrepancia encontrada
        'Pacu Frito': 'pacu',  # Ajuste basado en la discrepancia encontrada
        'Pejerrey': 'Pejerrey',
        'Picana': 'Picana',
        'Picante de Pollo': 'Picante_de_Pollo',
        'Pique Macho': 'pique',  # Ajuste basado en la discrepancia encontrada
        'Ranga': 'Ranga',
        'Saice': 'Saice',
        'Silpancho': 'silpancho',
        'Sopa de Maní': 'Sopa_de_Mani',  # Ajustado por caracteres especiales
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

def preparar_datos():
    """
    Prepara el dataset para entrenamiento
    """
    # Mapeo de nombres
    nombre_mapeo = crear_mapeo_nombres()
    
    # Directorio base
    data_dir = 'dataset_imagenes'
    
    # Crear lista de imágenes y etiquetas
    image_paths = []
    labels = []
    
    # Obtener clases posibles (basadas en carpetas existentes)
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_indices = {name: idx for idx, name in enumerate(classes)}
    
    # Recoger todas las imágenes y sus etiquetas
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.JPG')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_indices[class_name])
    
    print(f"Total de imágenes encontradas: {len(image_paths)}")
    print(f"Número de clases: {len(classes)}")
    print(f"Clases: {classes}")
    
    return image_paths, labels, classes

class ComidaClassifier(nn.Module):
    """
    Modelo de clasificación de comidas usando transfer learning con ResNet50
    """
    def __init__(self, num_classes):
        super(ComidaClassifier, self).__init__()
        # Cargar modelo preentrenado
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

def entrenar_modelo():
    """
    Entrena el modelo de clasificación de imágenes
    """
    print("Preparando datos...")
    image_paths, labels, classes = preparar_datos()
    
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
                'classes': classes
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
    
    print(f"Entrenamiento completado! Mejor precisión: {best_val_acc:.2f}%")
    
    return model, historial

def main():
    """
    Función principal
    """
    print("=== Entrenamiento del modelo de visión por computadora (PyTorch) ===")
    print("Este script entrena un modelo para clasificar comidas típicas bolivianas")
    print()
    
    # Verificar existencia de directorios y archivos
    if not os.path.exists('dataset_imagenes'):
        print("Error: No se encontró el directorio 'dataset_imagenes'")
        return
    
    if not os.path.exists('dataset_ingredientes.csv'):
        print("Error: No se encontró el archivo 'dataset_ingredientes.csv'")
        return
    
    if not os.path.exists('nutricion/datos_nutricionales.csv'):
        print("Error: No se encontró el archivo 'nutricion/datos_nutricionales.csv'")
        return
    
    print("Directorios y archivos verificados")
    print()
    
    # Iniciar entrenamiento
    try:
        model, historial = entrenar_modelo()
        print("Entrenamiento completado exitosamente!")
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
Script de prueba para verificar la carga del modelo entrenado
"""
import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# Añadir el directorio actual al path para importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar la clase modelo desde el archivo de entrenamiento
try:
    from entrenar_modelo_pytorch import ComidaClassifier
    print("[OK] ComidaClassifier importado exitosamente")
except Exception as e:
    print(f"[ERROR] Error al importar ComidaClassifier: {e}")
    sys.exit(1)

def probar_carga_modelo():
    # Verificar si existe el archivo modelo_completo.pth
    if not os.path.exists('modelo_completo.pth'):
        print("[ERROR] No se encontró el archivo modelo_completo.pth")
        return False
    
    print("[OK] Archivo modelo_completo.pth encontrado")
    
    # Cargar el checkpoint
    try:
        checkpoint = torch.load('modelo_completo.pth', map_location=torch.device('cpu'))
        print("[OK] Checkpoint cargado exitosamente")
    except Exception as e:
        print(f"[ERROR] Error al cargar checkpoint: {e}")
        return False
    
    # Verificar contenido del checkpoint
    print("Contenido del checkpoint:", checkpoint.keys())
    
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
        print(f"[OK] Clases encontradas: {len(classes)} clases")
        print(f"  Clases: {classes}")
    else:
        print("[ERROR] No se encontraron 'classes' en el checkpoint")
        return False
    
    if 'model_state_dict' in checkpoint:
        print("[OK] model_state_dict encontrado")
    else:
        print("[ERROR] No se encontró 'model_state_dict' en el checkpoint")
        return False
    
    # Intentar crear el modelo
    try:
        model = ComidaClassifier(num_classes=len(classes))
        print("[OK] Modelo creado exitosamente")
    except Exception as e:
        print(f"[ERROR] Error al crear modelo: {e}")
        return False
    
    # Cargar los pesos del modelo
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[OK] Pesos del modelo cargados exitosamente")
    except Exception as e:
        print(f"[ERROR] Error al cargar pesos del modelo: {e}")
        return False
    
    # Poner modelo en modo evaluación
    model.eval()
    print("[OK] Modelo en modo evaluación")
    
    print("\n[OK] Carga del modelo completada exitosamente!")
    return True

if __name__ == "__main__":
    print("Probando carga del modelo entrenado...")
    print("="*50)
    
    if probar_carga_modelo():
        print("\n[OK] ¡Éxito! El modelo se puede cargar correctamente.")
        print("El problema puede estar en cómo se llama la GUI o en otros aspectos del código.")
    else:
        print("\n[ERROR] Error en la carga del modelo.")
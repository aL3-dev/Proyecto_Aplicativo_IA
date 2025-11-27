"""
Script para cargar el modelo existente y reconstruirlo con la arquitectura correcta
"""
import torch
import torch.nn as nn
from torchvision import models
import sys

def crear_modelo_compatibilidad(num_classes):
    """
    Crea el modelo con la arquitectura que coincide con el modelo guardado
    Basado en el error de carga, parece que el modelo original tenía:
    - Primera capa intermedia con 512 unidades (no 128)
    - Posible capa BatchNorm (por los keys running_mean, running_var)
    """
    class ComidaClassifierOriginal(nn.Module):
        def __init__(self, num_classes):
            super(ComidaClassifierOriginal, self).__init__()
            self.resnet = models.resnet50(pretrained=False)
            
            # Congelar capas iniciales (como se hace en el entrenamiento original)
            for param in self.resnet.parameters():
                param.requires_grad = False
                
            # Reemplazar la capa final con la arquitectura adecuada
            # Basado en el error, parece que tiene BatchNorm en la capa 3
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),  # Originalmente 512, no 128
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),  # Esta capa puede estar presente
                nn.ReLU(),
                nn.Linear(128, 64),   # Y esta capa
                nn.ReLU(),
                nn.Linear(64, num_classes)  # Final
            )
    
        def forward(self, x):
            x = self.resnet(x)
            return x
    
    return ComidaClassifierOriginal(num_classes)

def crear_modelo_compatibilidad_actualizada(num_classes):
    """
    Versión del modelo que intenta emular la estructura original basada en los errores
    """
    class ComidaClassifier(nn.Module):
        def __init__(self, num_classes):
            super(ComidaClassifier, self).__init__()
            self.resnet = models.resnet50(pretrained=False)
            
            # Congelar capas iniciales
            for param in self.resnet.parameters():
                param.requires_grad = False
                
            # La arquitectura original parece tener esta estructura basada en los nombres de las capas:
            # resnet.fc.1.weight, resnet.fc.1.bias - capa Linear de num_features a 512
            # resnet.fc.3.weight, resnet.fc.3.bias, resnet.fc.3.running_mean, etc. - capa BatchNorm
            # resnet.fc.5.weight, resnet.fc.5.bias - capa Linear
            # resnet.fc.7.weight, resnet.fc.7.bias - capa Linear final
            
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),  # .1
                nn.ReLU(),                     # .2
                nn.Dropout(0.3),               # .3
                nn.Linear(512, 64),            # .4 - esta podría no existir o ser diferente
                nn.ReLU(),                     # .5
                nn.Linear(64, num_classes)     # .6 - esta podría no existir o ser diferente
            )
    
        def forward(self, x):
            x = self.resnet(x)
            return x
    
    return ComidaClassifier(num_classes)

def crear_modelo_version_3(num_classes):
    """
    Versión más precisa basada en los errores exactos
    """
    class ComidaClassifier(nn.Module):
        def __init__(self, num_classes):
            super(ComidaClassifier, self).__init__()
            self.resnet = models.resnet50(pretrained=False)

            # Congelar capas iniciales
            for param in self.resnet.parameters():
                param.requires_grad = False

            # Basado en los errores detectados:
            # size mismatch para resnet.fc.5.weight: shape torch.Size([256, 512]) en checkpoint vs torch.Size([64, 512]) actual
            # size mismatch para resnet.fc.7.weight: shape torch.Size([28, 256]) en checkpoint vs torch.Size([28, 64]) actual

            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.5),              # 0
                nn.Linear(num_features, 512), # 1 - .1.weight/.bias
                nn.ReLU(),                    # 2
                nn.BatchNorm1d(512),          # 3 - .3.weight/.bias/.running_mean/.running_var/.num_batches_tracked
                nn.Dropout(0.3),              # 4
                nn.Linear(512, 256),          # 5 - .5.weight/.bias (era 64 en el intento anterior)
                nn.ReLU(),                    # 6
                nn.Linear(256, num_classes)   # 7 - .7.weight/.bias  (era 64->num_classes en el intento anterior)
            )

        def forward(self, x):
            x = self.resnet(x)
            return x

    return ComidaClassifier(num_classes)

def probar_carga():
    # Cargar el checkpoint existente
    try:
        checkpoint = torch.load('modelo_completo.pth', map_location=torch.device('cpu'))
        print("[OK] Checkpoint cargado")
        print("Keys en el checkpoint:", list(checkpoint.keys()))
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el checkpoint: {e}")
        return False
    
    # Intentar crear modelo con la arquitectura adecuada
    try:
        num_classes = len(checkpoint['classes'])
        print(f"[INFO] Número de clases: {num_classes}")
        
        # Utilizar la versión 3 que incluye BatchNorm
        model = crear_modelo_version_3(num_classes)
        print("[OK] Modelo creado con arquitectura compatible")
        
        # Intentar cargar los pesos
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[OK] Pesos del modelo cargados exitosamente!")
        
        # Poner en modo evaluación
        model.eval()
        print("[OK] Modelo en modo evaluación")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error al crear o cargar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Probando diferentes arquitecturas de modelo para compatibilidad...")
    print("="*60)
    
    if probar_carga():
        print("\n[OK] ¡Éxito! El modelo es compatible con los pesos guardados.")
    else:
        print("\n[ERROR] No se pudo lograr compatibilidad con el modelo guardado.")
        print("Posibles soluciones:")
        print("- Reentrenar el modelo con la definición actual de ComidaClassifier")
        print("- O usar una versión específica del modelo que coincida con los pesos guardados")
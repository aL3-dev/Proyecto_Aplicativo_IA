# Instrucciones de Instalación para el Sistema de Reconocimiento de Comidas Típicas Bolivianas

## Solución de Problemas de Versiones

Si tienes problemas con las versiones de las bibliotecas, aquí te proporcionamos diferentes métodos de instalación:

### Método 1: Instalación con pip (recomendado para CPU)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate

# Instalar PyTorch con soporte CPU solamente
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Instalar las demás dependencias
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 Pillow>=8.3.0 scikit-learn>=1.0.0 tqdm>=4.60.0
```

### Método 2: Instalación con GPU (si tienes tarjeta NVIDIA compatible)

Primero, verifica la versión de CUDA instalada en tu sistema con el comando:
```bash
nvidia-smi
```

Luego, instala las bibliotecas correspondientes a tu versión de CUDA:

#### Para CUDA 12.6:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 Pillow>=8.3.0 scikit-learn>=1.0.0 tqdm>=4.60.0
```

#### Para CUDA 12.8:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 Pillow>=8.3.0 scikit-learn>=1.0.0 tqdm>=4.60.0
```

### Método 3: Instalación alternativa con conda (si pip no funciona)

```bash
# Si usas Anaconda o Miniconda
conda install pytorch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 pytorch-cuda=12.6 -c pytorch -c nvidia
conda install pandas numpy matplotlib pillow scikit-learn tqdm
```

### Método 4: Instalación sin especificar versiones (última opción)

Si todas las anteriores fallan, puedes intentar instalar las últimas versiones disponibles:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib Pillow scikit-learn tqdm
```

## Compatibilidad de Python

Asegúrate de usar Python 3.8 o superior para compatibilidad con las bibliotecas requeridas.

## Validación de la Instalación

Después de instalar las bibliotecas, puedes verificar la instalación con este script simple:

```python
# test_instalacion.py
try:
    import torch
    print(f"PyTorch instalado: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    import torchvision
    print(f"Torchvision instalado: {torchvision.__version__}")
    
    import pandas as pd
    print(f"Pandas instalado: {pd.__version__}")
    
    import numpy as np
    print(f"NumPy instalado: {np.__version__}")
    
    import matplotlib
    print(f"Matplotlib instalado: {matplotlib.__version__}")
    
    import PIL
    print(f"Pillow instalado: {PIL.__version__}")
    
    import sklearn
    print(f"Scikit-learn instalado: {sklearn.__version__}")
    
    import tqdm
    print(f"Tqdm instalado: {tqdm.__version__}")
    
    print("¡Todas las bibliotecas se instalaron correctamente!")
    
except ImportError as e:
    print(f"Error al importar biblioteca: {e}")
```

Ejecuta este script para verificar que todas las bibliotecas estén instaladas correctamente:

```bash
python test_instalacion.py
```

## Solución de Problemas Comunes

### Error de compatibilidad de CUDA
- Verifica tu versión de CUDA con `nvidia-smi`
- Instala la versión de PyTorch que coincida con tu versión de CUDA

### Error de permisos en Windows
- Ejecuta el símbolo del sistema como administrador
- O usa el parámetro `--user`: `pip install --user -r requirements.txt`

### Espacio insuficiente
- Asegúrate de tener al menos 4-5 GB libres
- PyTorch puede ocupar considerable espacio de disco

### Si aún tienes problemas
1. Desinstala versiones anteriores: `pip uninstall torch torchvision torchaudio`
2. Limpia el caché de pip: `pip cache purge`
3. Vuelve a instalar con uno de los métodos anteriores
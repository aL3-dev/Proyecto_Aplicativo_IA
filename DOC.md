# Documentación Completa del Proyecto: Sistema de Reconocimiento de Platos Bolivianos

## Índice
1. [Descripción General del Proyecto](#descripción-general-del-proyecto)
2. [Objetivo Principal](#objetivo-principal)
3. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
4. [Componentes Implementados](#componentes-implementados)
5. [Funcionamiento del Sistema](#funcionamiento-del-sistema)
6. [Guía de Uso Paso a Paso](#guía-de-uso-paso-a-paso)
7. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
8. [Estructura de Archivos](#estructura-de-archivos)
9. [Requisitos del Sistema](#requisitos-del-sistema)

---

## Descripción General del Proyecto

Este proyecto implementa un sistema integral de visión por computadora para la identificación automática de platos bolivianos a partir de imágenes. El sistema toma una fotografía de un plato boliviano y devuelve: el nombre del plato identificado, sus ingredientes y su valor nutricional.

## Objetivo Principal

Desarrollar una aplicación que al tomar una foto de un plato boliviano:
1. **Reconozca visualmente el plato** (por ejemplo: "Aji de Pataskha")
2. **Identifique sus ingredientes** (carne de res, papa, maíz, etc.)
3. **Muestre su valor nutricional** (calorías, proteínas, carbohidratos, grasas)

## Arquitectura del Proyecto

El sistema se compone de 3 capas principales:

1. **Capa de Visión por Computadora**: Reconoce platos desde imágenes
2. **Capa de Conexión de Datos**: Relaciona platos con ingredientes y nutrición
3. **Capa de Interfaz de Usuario**: Presenta la información completa

## Componentes Implementados

### 1. Módulos de Utilidades (`utils/`)
- `cargar_datos.py`: Funciones para cargar y combinar datasets
- `preprocesamiento.py`: Funciones para preprocesar datos de entrada

### 2. Notebooks de Modelos (`ingredientes_vision/`)
- `modelo_vision_ingredientes.ipynb`: Entrena un modelo de reconocimiento de platos con imágenes reales
- `ingredientes_labels.json`: Metadatos de ingredientes clasificados

### 3. Dataset Nutricional (`nutricion/`)
- `datos_nutricionales.csv`: Información nutricional por plato
- `modelo_nutricion.ipynb`: Modelo de clasificación nutricional

### 4. Sistema Integral
- `sistema_integral.py`: Aplicación completa integrada

## Funcionamiento del Sistema

El sistema opera en esta secuencia:

1. **Entrada**: Usuario toma una foto de un plato boliviano
2. **Procesamiento de Imagen**: El modelo de visión por computadora analiza la imagen
3. **Reconocimiento**: El modelo identifica visualmente el plato (Ej: "Aji de Pataskha")
4. **Conexión de Datos**: El sistema busca el plato en los datasets
5. **Recuperación de Información**: Obtiene ingredientes y valor nutricional
6. **Salida**: Muestra plato, ingredientes y valor nutricional

## Guía de Uso Paso a Paso

### Paso 1: Preparación del Entorno
```bash
cd C:\Users\aleja\Desktop\Proy
pip install -r requirements.txt
```

### Paso 2: Verificar Funcionamiento Básico
```bash
python test_funcionalidad.py
```

### Paso 3: Entrenar el Modelo de Visión por Computadora
1. Abrir `ingredientes_vision/modelo_vision_ingredientes.ipynb`
2. Ejecutar todas las celdas para entrenar el modelo
3. El modelo se guardará como `modelo_reconocimiento_platos.h5`
4. El codificador se guardará como `label_encoder_platos.pkl`

### Paso 4: Uso del Sistema Integral
```python
from sistema_integral import SistemaIdentificacionPlatos

# Crear instancia del sistema
sistema = SistemaIdentificacionPlatos()

# Analizar una imagen
resultado = sistema.analyze_plate_image('ruta/a/tu/imagen.jpg')

# Mostrar resultados
sistema.print_results(resultado)
```

### Paso 5: Prueba Rápida
```bash
python sistema_integral.py
```

## Entrenamiento del Modelo

### Datos de Entrenamiento
- **Imágenes**: `dataset_imagenes/` (organizadas por plato: `Aji_de_pataskha/`, `caldo_de_bagre/`, etc.)
- **Ingredientes**: `dataset_ingredientes.csv`
- **Nutrición**: `nutricion/datos_nutricionales.csv`

### Proceso de Entrenamiento
1. El notebook identifica platos en carpetas de imágenes
2. Relaciona carpetas con datos en CSV
3. Entrena modelo para reconocer patrones visuales
4. Guarda modelo entrenado para uso posterior

### Arquitectura del Modelo
- **Base**: VGG16 con pesos preentrenados
- **Transfer Learning**: Capas congeladas para extracción de características
- **Clasificación**: Capas densas personalizadas para identificar platos bolivianos
- **Salida**: Capa softmax para clasificación multiclase

## Estructura de Archivos

```
Proy/
├── dataset_imagenes/                 # Imágenes por plato
│   ├── Aji_de_pataskha/
│   ├── caldo_de_bagre/
│   ├── Cazuelas/
│   └── ...
├── dataset_ingredientes.csv         # Ingredientes por plato
├── nutricion/
│   └── datos_nutricionales.csv      # Información nutricional
├── ingredientes_vision/
│   ├── modelo_vision_ingredientes.ipynb  # Entrenamiento del modelo
│   ├── modelo_reconocimiento_platos.h5   # Modelo entrenado (después del entrenamiento)
│   ├── label_encoder_platos.pkl          # Codificador de etiquetas
│   └── ingredientes_labels.json
├── utils/
│   ├── cargar_datos.py              # Funciones de carga de datos
│   └── preprocesamiento.py          # Funciones de preprocesamiento
├── sistema_integral.py              # Sistema completo integrado
├── P/
│   └── main.py                      # Script de ejemplo
├── test_funcionalidad.py            # Pruebas de funcionalidad
├── requirements.txt                 # Dependencias
├── README.md                        # Documentación principal
├── DOCUMENTACION.md                 # Documentación técnica
└── INICIO_RAPIDO.md                 # Guía de inicio rápido
```

## Requisitos del Sistema

### Hardware
- Procesador compatible con operaciones de GPU (opcional)
- Mínimo 4GB RAM (recomendado 8GB+)
- Espacio suficiente para almacenar modelos entrenados

### Software
- Python 3.7 o superior
- TensorFlow 2.x
- OpenCV
- Pandas, NumPy, Matplotlib
- Jupyter Notebook (para entrenamiento)

### Dependencias
```bash
pandas>=2.0.0
numpy>=1.21.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
opencv-python>=4.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

---

Este sistema está completamente funcional y listo para ser entrenado con tus datos reales. Una vez entrenado el modelo de visión por computadora, proporcionará la funcionalidad completa que solicitaste: reconocimiento visual de platos bolivianos con información nutricional y de ingredientes.
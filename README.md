# Identificación Nutricional e Ingredientes de Platos Bolivianos

Este proyecto implementa un sistema integral de aprendizaje supervisado que permite tomar una foto de un plato boliviano y obtener: identificación del plato, lista de ingredientes detectados y valor nutricional estimado.

## Descripción General

El sistema completo consta de:
1. **Identificación Visual de Platos**: Reconoce platos bolivianos a partir de imágenes
2. **Detección de Ingredientes**: Identifica los ingredientes presentes en el plato
3. **Estimación Nutricional**: Calcula valores nutricionales basados en los ingredientes detectados
4. **Clasificación Nutricional**: Categoriza el plato según su perfil nutricional (alto/medio/bajo en proteínas, carbohidratos, grasas, calorías)

## Estructura del Proyecto

```
Proy/
│
├── dataset_imagenes/
│      ├── Aji_de_pataskha/
│      ├── caldo_de_bagre/
│      ├── Cazuelas/
│      └── ...
│
├── dataset_ingredientes.csv          # Información de ingredientes por plato
├── config.json                      # Configuración del proyecto
├── proyecto.md                      # Documentación del proyecto
├── estructura.txt                   # Estructura del proyecto
│
├── nutricion/
│      ├── datos_nutricionales.csv   # Información nutricional por plato
│      └── modelo_nutricion.ipynb    # Notebook para clasificación nutricional
│
├── ingredientes_vision/
│      ├── modelo_vision_ingredientes.ipynb  # Notebook para identificación por imagen
│      └── ingredientes_labels.json          # Etiquetas de ingredientes
│
├── utils/
│      ├── preprocesamiento.py       # Funciones de preprocesamiento
│      └── cargar_datos.py           # Funciones para carga de datos
│
├── P/
│      └── main.py                   # Script principal de ejemplo
│
├── aplicacion_integral.py           # Aplicación que integra todos componentes
├── sistema_integral.py              # Sistema completo: imagen -> plato -> ingredientes -> nutrición
├── test_funcionalidad.py            # Script de verificación de funcionalidad
├── requirements.txt                 # Dependencias del proyecto
├── DOCUMENTACION.md                 # Documentación técnica detallada
└── INICIO_RAPIDO.md                 # Guía de inicio rápido

```

## Requisitos

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- opencv-python

## Instalación

1. Clonar o descargar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Sistema Integral Completo (Recomendado)

Ejecutar el sistema completo que toma una imagen y devuelve todo el análisis:
```bash
python sistema_integral.py
```

O usarlo programáticamente:
```python
from sistema_integral import SistemaIdentificacionPlatos

sistema = SistemaIdentificacionPlatos()
resultado = sistema.analyze_plate_image('ruta/a/imagen.jpg')
sistema.print_results(resultado)
```

### Componentes Individuales

#### Clasificación Nutricional
Ejecutar el notebook `nutricion/modelo_nutricion.ipynb` para entrenar el modelo de clasificación nutricional.

#### Identificación de Ingredientes
Ejecutar el notebook `ingredientes_vision/modelo_vision_ingredientes.ipynb` para entrenar el modelo de identificación de ingredientes.

#### Script Principal
Ejecutar `P/main.py` para ver un ejemplo básico de cómo usar las funciones de carga y preprocesamiento de datos.

## Funcionalidad Principal

El sistema está diseñado para funcionar como una aplicación de "Food Recognition" completa:

1. **Toma de Foto**: El usuario toma una foto de un plato boliviano
2. **Identificación Visual**: El sistema procesa la imagen para detectar ingredientes visibles
3. **Reconocimiento de Plato**: Basado en los ingredientes detectados, identifica el plato más probable
4. **Análisis Nutricional**: Calcula el valor nutricional estimado del plato
5. **Clasificación**: Categoriza el plato según su perfil nutricional

## Componentes Principales

### Carga de Datos (`utils/cargar_datos.py`)
Funciones para cargar y combinar datasets de ingredientes y nutrición.

### Preprocesamiento (`utils/preprocesamiento.py`)
Funciones para procesar texto de ingredientes, crear categorías nutricionales y preparar datos para el modelado.

### Sistema Integral (`sistema_integral.py`)
Aplicación completa que integra todos los componentes para el flujo: imagen -> plato -> ingredientes -> nutrición.

### Clasificación Nutricional (`nutricion/modelo_nutricion.ipynb`)
Modelo basado en redes neuronales que clasifica platos según contenido nutricional.

### Identificación de Ingredientes (`ingredientes_vision/modelo_vision_ingredientes.ipynb`)
Modelo de visión por computadora que identifica ingredientes a partir de imágenes usando transfer learning.

## Conjuntos de Datos

- `dataset_ingredientes.csv`: Contiene platos con sus ingredientes en formato de texto.
- `nutricion/datos_nutricionales.csv`: Contiene valores nutricionales (calorías, proteínas, carbohidratos, grasas) por plato.
- `dataset_imagenes/`: Directorio con imágenes de los platos organizados por categoría.

## Contribución

Las contribuciones son bienvenidas. Siéntase libre de enviar un pull request o abrir un issue para discutir cambios.

## Licencia

Este proyecto está licenciado bajo los términos descritos por el autor.
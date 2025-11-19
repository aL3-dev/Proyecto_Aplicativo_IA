# Documentación del Proyecto: Identificación Nutricional e Ingredientes de Platos Bolivianos

## Descripción del Proyecto

Este proyecto implementa modelos de aprendizaje supervisado para:
1. Clasificación nutricional de platos bolivianos (alto/medio/bajo en proteínas, carbohidratos, grasas, calorías)
2. Identificación de ingredientes en platos bolivianos a partir de imágenes

## Estructura del Proyecto

- `dataset_ingredientes.csv`: Información de ingredientes por plato
- `nutricion/datos_nutricionales.csv`: Datos nutricionales por plato
- `dataset_imagenes/`: Imágenes de platos organizadas por directorio
- `utils/cargar_datos.py`: Funciones para cargar y procesar datos
- `utils/preprocesamiento.py`: Funciones de preprocesamiento
- `nutricion/modelo_nutricion.ipynb`: Notebook para modelo de clasificación nutricional
- `ingredientes_vision/modelo_vision_ingredientes.ipynb`: Notebook para modelo de identificación de ingredientes

## Módulos Desarrollados

### 1. Módulo `cargar_datos.py`

Funciones principales:
- `cargar_datos_ingredientes()`: Carga el dataset de ingredientes
- `cargar_datos_nutricion()`: Carga el dataset nutricional
- `procesar_ingredientes()`: Procesa la cadena de ingredientes a lista
- `combinar_datos()`: Combina ambos datasets
- `obtener_ruta_imagenes()`: Obtiene rutas de imágenes por plato

### 2. Módulo `preprocesamiento.py`

Funciones principales:
- `preprocesar_texto_ingredientes()`: Convierte lista de ingredientes a texto
- `crear_etiquetas_nutricionales()`: Crea categorías nutricionales basadas en percentiles
- `tokenizar_ingredientes()`: Tokeniza texto de ingredientes para modelos de ML
- `escalar_datos_nutricionales()`: Escala datos numéricos nutricionales
- `codificar_etiquetas()`: Codifica etiquetas categóricas para modelos

## Modelos Implementados

### 1. Modelo de Clasificación Nutricional

- Tipo: Red neuronal con capas densas y LSTM
- Entrada: Texto de ingredientes tokenizado
- Salida: Categoría nutricional (alto/medio/bajo en proteínas, carbohidratos, grasas, calorías)
- Métricas: Accuracy, precision, recall, F1-score

### 2. Modelo de Identificación de Ingredientes

- Tipo: Red neuronal convolucional con transfer learning (VGG16)
- Entrada: Imágenes de platos
- Salida: Vector binario de ingredientes presentes (multilabel)
- Métricas: Hamming loss, accuracy, classification report

## Uso del Proyecto

1. Instalar dependencias: `pip install -r requirements.txt`
2. Verificar la funcionalidad: `python test_funcionalidad.py`
3. Ejecutar modelos:
   - Clasificación nutricional: `nutricion/modelo_nutricion.ipynb`
   - Identificación de ingredientes: `ingredientes_vision/modelo_vision_ingredientes.ipynb`
4. Ejecutar script principal: `P/main.py`

## Características Técnicas

- Aprendizaje supervisado
- Procesamiento de lenguaje natural (NLP)
- Visión por computadora
- Transfer learning
- Modelos de clasificación multietiqueta
- Preprocesamiento de datos

## Resultados Esperados

- Capacidad para clasificar platos según su perfil nutricional
- Identificación de ingredientes en imágenes de platos bolivianos
- Contribución a herramientas de análisis nutricional accesible
- Apoyo a la preservación gastronómica boliviana
- Aplicación potencial en salud, gastronomía y educación alimentaria
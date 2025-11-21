# Detalles del Proyecto: Sistema de Reconocimiento de Comidas Típicas Bolivianas

## Descripción General
Proyecto de visión por computadora que entrena un modelo para reconocer comidas típicas bolivianas a partir de imágenes, y proporciona información nutricional e ingredientes del plato identificado.

## Componentes del Proyecto

### 1. Dataset de Imágenes
- **Cantidad**: 27 platos bolivianos diferentes
- **Estructura**: Cada plato tiene su propia carpeta con 50-100+ imágenes
- **Directorio**: `dataset_imagenes/`
- **Formato**: Imágenes en formato JPEG
- **Ejemplo de estructura**:
  ```
  dataset_imagenes/
  ├── Aji_de_pataskha/ (contiene 100+ imágenes)
  ├── caldo_de_bagre/ (contiene 50+ imágenes)
  ├── Chairo/ (contiene 100+ imágenes)
  └── ... (otros 24 platos)
  ```

### 2. Dataset de Ingredientes
- **Archivo**: `dataset_ingredientes.csv`
- **Número de registros**: 27 platos
- **Estructura**: 
  - Columna 1: `plato` - Nombre del plato
  - Columna 2: `ingredientes` - Lista de ingredientes en formato array
- **Ejemplo**: 
  ```
  Aji de Pataskha,"['carne de res', 'maíz pelado', 'cabeza de cerdo', 'cebolla blanca', 'cebolla verde', 'apio', 'comino']"
  ```

### 3. Dataset de Información Nutricional
- **Archivo**: `nutricion/datos_nutricionales.csv`
- **Directorio**: `nutricion/`
- **Número de registros**: 27 platos
- **Estructura**:
  - Columna 1: `plato` - Nombre del plato
  - Columna 2: `calorias` - Calorías totales
  - Columna 3: `proteina` - Proteína en gramos
  - Columna 4: `carbohidratos` - Carbohidratos en gramos
  - Columna 5: `grasa` - Grasa en gramos
- **Ejemplo**:
  ```
  Aji de Pataskha,480,32,38,22
  ```

## Funcionalidad del Sistema

### Flujo Principal
1. El usuario carga una imagen de un plato de comida boliviano
2. El modelo de visión por computadora identifica qué plato es
3. El sistema recupera los ingredientes y la información nutricional del plato identificado
4. Se muestra al usuario la lista de ingredientes y los valores nutricionales

### Ejemplo de Funcionamiento
**Entrada**: Imagen de Aji de Pataskha
**Procesamiento**:
- Sistema identifica: "Aji de Pataskha"
- Recupera ingredientes: carne de res, maíz pelado, cabeza de cerdo, cebolla blanca, cebolla verde, apio, comino
- Recupera valores nutricionales: calorias = 480, proteina = 32, carbohidratos = 38, grasa = 22
**Salida**: Muestra nombre del plato, ingredientes y valores nutricionales

## Tecnologías y Herramientas Requeridas

### Lenguaje de Programación
- Python 3.8+

### Bibliotecas Principales
- TensorFlow o PyTorch (para el modelo de visión por computadora)
- OpenCV (para procesamiento de imágenes)
- pandas (para manejo de datasets)
- scikit-learn (para utilidades de machine learning)
- Flask o FastAPI (para crear la API web)

### Entorno de Desarrollo
- Jupyter Notebooks (para experimentación y desarrollo)
- IDE como PyCharm o Visual Studio Code

### Hardware Recomendado
- GPU opcional pero recomendada para entrenamiento acelerado
- Mínimo 8GB de RAM (16GB recomendado)

## Plan de Implementación

### Fase 1: Preparación de Datos
1. **Limpieza de imágenes**:
   - Verificar consistencia en los nombres de carpetas de imágenes vs. nombres en CSV
   - Eliminar imágenes duplicadas o irrelevantes
   - Verificar que todas las imágenes sean válidas
2. **Preprocesamiento**:
   - Redimensionar imágenes a tamaño uniforme (por ejemplo, 224x224)
   - Normalizar valores de píxeles
   - Dividir dataset en entrenamiento (70%), validación (15%) y prueba (15%)

### Fase 2: Entrenamiento del Modelo
1. **Selección del modelo**:
   - Usar transfer learning con modelos preentrenados (ResNet, VGG, EfficientNet)
2. **Entrenamiento**:
   - Configurar hiperparámetros (learning rate, batch size, epochs)
   - Entrenar el modelo con el dataset de imágenes
   - Validar periódicamente para evitar overfitting
3. **Evaluación**:
   - Probar el modelo con el dataset de prueba
   - Calcular métricas de rendimiento (accuracy, precision, recall, F1-score)

### Fase 3: Integración de Datos
1. **Conexión de datasets**:
   - Crear sistema de búsqueda para conectar resultados del modelo con información nutricional e ingredientes
   - Validar que todos los platos reconoció el modelo tengan información disponible en los CSV

### Fase 4: Desarrollo de la Aplicación
1. **Backend**:
   - Crear API con endpoints para subir imágenes y recibir resultados
   - Implementar lógica para procesar imágenes y devolver información
2. **Frontend** (opcional para MVP):
   - Interfaz simple para subir imágenes
   - Visualización de resultados (nombre del plato, ingredientes, valores nutricionales)

## Platos Disponibles en el Dataset

### Lista Completa de Platos (27 en total)
1. Aji de Pataskha
2. Caldo de Bagre
3. Cazuela
4. Chairo
5. Charquekan Orureño
6. Chicharrón
7. Chorizos
8. Cunape
9. Empanada de Arroz
10. Falso Conejo
11. Fricase
12. Fritanga
13. Karapecho
14. Keperi Beniano
15. La Kalapurka
16. Locro
17. Majao
18. Mondongo Chuquisaqueño
19. Pacu Frito
20. Pejerrey
21. Picana
22. Picante de Pollo
23. Pique Macho
24. Ranga
25. Saice
26. Silpancho
27. Sopa de Maní
28. Sopa de Quinua

## Consideraciones Técnicas

### Desafíos Potenciales
1. **Variabilidad de imágenes**: Diferentes ángulos, iluminación y presentación pueden afectar el rendimiento
2. **Consistencia de nombres**: Verificar que los nombres en las carpetas de imágenes coincidan exactamente con los nombres en los CSV
3. **Calidad de imágenes**: Algunas imágenes pueden estar duplicadas o no representar correctamente el plato

### Optimizaciones Posibles
1. **Data augmentation** para aumentar la robustez del modelo
2. **Fine-tuning** del modelo preentrenado para mejorar el rendimiento
3. **Técnicas de regularización** para evitar overfitting

## Resultados Esperados

### Funcionalidad
- Capacidad de reconocer correctamente los 27 platos bolivianos con alta precisión
- Integración fluida entre la clasificación de imagen y la información nutricional/ingredientes
- Interfaz amigable para la interacción del usuario

### Métricas de Éxito
- Precisión del modelo superior al 85%
- Tiempo de respuesta inferior a 5 segundos por imagen
- 100% de cobertura de los platos en el dataset de información nutricional

## Recomendaciones para Implementación
1. Comenzar con un modelo simple de transfer learning
2. Validar la consistencia de nombres entre datasets antes de entrenar
3. Realizar pruebas iterativas para mejorar el modelo
4. Considerar la posibilidad de desplegar en la nube para mayor accesibilidad
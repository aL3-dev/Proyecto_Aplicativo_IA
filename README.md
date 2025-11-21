# Sistema de Reconocimiento de Comidas Típicas Bolivianas

Este proyecto implementa un sistema de visión por computadora que puede identificar comidas típicas bolivianas a partir de imágenes y proporcionar información sobre sus ingredientes y valores nutricionales.

## Descripción del Proyecto

El sistema consta de:

1. **Modelo de visión por computadora**: Entrenado para reconocer 27 comidas típicas bolivianas
2. **Base de datos de ingredientes**: Información sobre los ingredientes de cada plato
3. **Base de datos nutricional**: Información sobre calorías, proteínas, carbohidratos y grasas

## Estructura del Proyecto

```
├── dataset_ingredientes.csv            # Lista de ingredientes por plato
├── nutricion/
│   └── datos_nutricionales.csv         # Información nutricional por plato
├── dataset_imagenes/                  # Imágenes organizadas por carpetas
│   ├── Aji_de_pataskha/
│   ├── caldo_de_bagre/
│   ├── ...
├── sistema_completo_pytorch.py        # Script principal del sistema (PyTorch)
├── entrenar_modelo_pytorch.py         # Script para entrenar el modelo (PyTorch)
├── predecir_plato_pytorch.py          # Script para predecir platos (PyTorch)
├── sistema_completo.py                # Versión original con TensorFlow
├── entrenar_modelo.py                 # Versión original con TensorFlow
├── predecir_plato.py                  # Versión original con TensorFlow
└── verificar_concordancia.py          # Script para verificar nombres
```

## Instalación de Dependencias

Para ejecutar este proyecto con PyTorch, necesitas instalar las siguientes bibliotecas Python:

```bash
pip install torch torchvision torchaudio pandas numpy matplotlib pillow scikit-learn tqdm
```

Si prefieres usar la versión original con TensorFlow:
```bash
pip install tensorflow pandas numpy matplotlib pillow scikit-learn
```

## Uso del Sistema (PyTorch)

### 1. Entrenamiento del Modelo

Ejecuta el script principal para entrenar el modelo con PyTorch:

```bash
python sistema_completo_pytorch.py
```

O directamente:
```bash
python entrenar_modelo_pytorch.py
```

Selecciona la opción 1 para entrenar el modelo. El proceso incluirá:
- Preprocesamiento de imágenes
- Entrenamiento con transfer learning usando ResNet50
- Validación y selección del mejor modelo
- Guardado del modelo entrenado

### 2. Clasificación de Imágenes

Después del entrenamiento, selecciona la opción 2 para clasificar una imagen:
- Carga una imagen de un plato boliviano
- El modelo identificará el plato y mostrará:
  - Nombre del plato con su nivel de confianza
  - Lista de ingredientes
  - Información nutricional (calorías, proteínas, carbohidratos, grasas)

## Detalles Técnicos (PyTorch)

### Arquitectura del Modelo

- Base: ResNet50 pre-entrenado en ImageNet
- Capas adicionales para clasificación específica
- Transfer learning con capas congeladas inicialmente
- Fine-tuning con capas descongeladas en etapas posteriores

### Preprocesamiento de Imágenes

- Tamaño estándar: 224x224 píxeles
- Normalización con valores ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Data augmentation durante el entrenamiento:
  - Rotación aleatoria
  - Volteo horizontal
  - Ajuste de color (brightness, contrast, saturation)
  - Jitter de color

### Manejo de Nombres Inconsistentes

El sistema incluye un mecanismo para resolver discrepancias entre los nombres de los platos en los archivos CSV y las carpetas de imágenes.

## Platos Soportados

El sistema puede reconocer los siguientes 27 platos:

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

## Ejemplo de Funcionamiento

**Entrada**: Imagen de Aji de Pataskha
**Procesamiento**:
- Sistema identifica: "Aji de Pataskha" con 95% de confianza
- Recupera ingredientes: carne de res, maíz pelado, cabeza de cerdo, cebolla blanca, cebolla verde, apio, comino
- Recupera valores nutricionales: calorias = 480, proteina = 32, carbohidratos = 38, grasa = 22
**Salida**: Muestra nombre del plato, ingredientes y valores nutricionales

## Archivos Generados

Después del entrenamiento con PyTorch, se generan los siguientes archivos:
- `modelo_completo.pth`: Modelo entrenado completo
- `mejor_modelo.pth`: Mejor modelo según métricas de validación
- `historial_entrenamiento.pkl`: Historial del entrenamiento
- `graficas_entrenamiento.png`: Gráficas de precisión y pérdida

## Comparación TensorFlow vs PyTorch

Este proyecto ofrece dos implementaciones:
- **PyTorch**: Más flexible y moderno, ideal para investigación y desarrollo
- **TensorFlow**: Más estable para despliegue en producción

Ambas implementaciones tienen la misma funcionalidad y objetivos.

## Contribuciones

Las contribuciones son bienvenidas. Para cambios mayores, por favor abre un issue primero para discutir qué te gustaría cambiar.
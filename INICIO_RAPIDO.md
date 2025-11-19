# Guía de Inicio Rápido

## Instalación

1. Clona o descarga este repositorio
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Verificación Rápida

Ejecuta el script de prueba para verificar que todo funciona correctamente:
```bash
python test_funcionalidad.py
```

## Estructura de Directorios

- `dataset_ingredientes.csv` - Lista de platos con sus ingredientes
- `nutricion/datos_nutricionales.csv` - Información nutricional por plato
- `dataset_imagenes/` - Imágenes organizadas por directorio de plato
- `utils/` - Módulos de utilidades para carga y preprocesamiento
- `nutricion/modelo_nutricion.ipynb` - Modelo de clasificación nutricional
- `ingredientes_vision/modelo_vision_ingredientes.ipynb` - Modelo de identificación de ingredientes
- `P/main.py` - Script de ejemplo de uso

## Ejecutar el Script Principal

```bash
python P/main.py
```

## Modelos Disponibles

### 1. Clasificación Nutricional
Abre y ejecuta `nutricion/modelo_nutricion.ipynb` para entrenar un modelo que clasifica platos según su contenido nutricional (alto/medio/bajo en proteínas, carbohidratos, grasas, calorías).

### 2. Identificación de Ingredientes
Abre y ejecuta `ingredientes_vision/modelo_vision_ingredientes.ipynb` para entrenar un modelo que identifica ingredientes en imágenes de platos bolivianos.

## Personalización

Puedes adaptar los modelos para:
- Ajustar las categorías nutricionales
- Cambiar los parámetros del modelo
- Agregar más datos de entrenamiento
- Mejorar la arquitectura del modelo
- Ajustar el preprocesamiento de datos

## Recursos

- Documentación completa: `DOCUMENTACION.md`
- Archivo de configuración: `config.json`
- Descripción del proyecto: `proyecto.md`
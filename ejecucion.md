# Orden de Ejecución del Proyecto: Identificación Nutricional e Ingredientes de Platos Bolivianos

Este archivo describe el orden correcto para ejecutar los archivos de código del proyecto para evitar fallos y asegurar un funcionamiento correcto.

## 1. Verificación Inicial
Antes de ejecutar cualquier componente principal, asegúrate de tener instaladas las dependencias:

```bash
pip install -r requirements.txt
```

## 2. Verificación del Funcionamiento Básico
**Ejecutar primero:** `test_funcionalidad.py`
```bash
python test_funcionalidad.py
```
Este script verifica que los módulos de utilidad funcionan correctamente y que los datasets están accesibles.

## 3. Componentes de Utilidad (No se ejecutan directamente)
Los siguientes archivos contienen funciones de utilidad y no deben ejecutarse individualmente:
- `utils/cargar_datos.py`
- `utils/preprocesamiento.py`

## 4. Script de Ejemplo Básico
**Ejecutar después:** `P/main.py`
```bash
python P/main.py
```
Este script demuestra cómo usar las funciones de carga y preprocesamiento de datos básicos.

## 5. Aplicación Integral (Opción 1)
**Ejecutar para ver la aplicación completa:** `aplicacion_integral.py`
```bash
python aplicacion_integral.py
```
Esta aplicación integra todos los componentes para una funcionalidad completa (identificación de plato a partir de ingredientes predichos).

## 6. Sistema Integral Completo (Opción 2) - Recomendado
**Ejecutar para el sistema completo:** `sistema_integral.py`
```bash
python sistema_integral.py
```
Este es el sistema más completo que toma una imagen y devuelve: identificación del plato, ingredientes detectados y valor nutricional estimado.

## 7. Notebooks de Entrenamiento (Opcional)
Los siguientes notebooks se utilizan para entrenar modelos y deben ejecutarse en entorno Jupyter:

### 7.1 Modelo de Clasificación Nutricional
`nutricion/modelo_nutricion.ipynb`
- Entrena el modelo de clasificación nutricional
- Clasifica platos según contenido nutricional (alto/medio/bajo en proteínas, carbohidratos, grasas, calorías)

### 7.2 Modelo de Identificación de Ingredientes
`ingredientes_vision/modelo_vision_ingredientes.ipynb`
- Entrena el modelo de identificación de ingredientes a partir de imágenes
- Utiliza redes neuronales convolucionales con transfer learning

## 8. Notas Importantes

- **Orden Crítico:** Siempre ejecuta `test_funcionalidad.py` antes que otras aplicaciones para verificar la funcionalidad básica.
- **Dependencias:** Los scripts `aplicacion_integral.py` y `sistema_integral.py` dependen de las funciones en `utils/`.
- **Archivos de Datos:** Asegúrate de que los archivos `dataset_ingredientes.csv` y `nutricion/datos_nutricionales.csv` existen antes de ejecutar cualquiera de las aplicaciones principales.
- **Directorios de Imágenes:** El sistema completo espera imágenes en el directorio `dataset_imagenes/` organizadas por tipo de plato.
- **Modelos Entrenados:** Si existen modelos previamente entrenados en `ingredientes_vision/modelo_reconocimiento_platos.h5` y `ingredientes_vision/label_encoder_platos.pkl`, `sistema_integral.py` los utilizará. Si no existen, usará una función simulada.
- **Problemas de Importación:** Si tienes problemas con importaciones, asegúrate de que los archivos han sido actualizados para usar la ruta correcta a los módulos en el directorio `utils/`. Los scripts principales han sido corregidos para importar correctamente desde `utils.cargar_datos` y `utils.preprocesamiento`.

## 9. Flujo Recomendado para Nuevo Usuario

1. `pip install -r requirements.txt`
2. `python test_funcionalidad.py` (verificación de funcionalidad)
3. `python P/main.py` (ejemplo básico)
4. `python sistema_integral.py` (sistema completo)
5. Opcional: `python aplicacion_integral.py` (alternativa al sistema completo)
6. Opcional: Ejecutar los notebooks en entorno Jupyter para entrenar modelos

## 10. Solución de Problemas Comunes

Si encuentras errores de tipo "ModuleNotFoundError" o problemas con las importaciones:
- Verifica que el directorio `utils/` esté en la misma carpeta que los scripts principales
- Asegúrate de que los archivos `utils/cargar_datos.py` y `utils/preprocesamiento.py` existen
- Los scripts principales han sido actualizados para usar las rutas de importación correctas

Si tienes problemas al cargar los datasets:
- Asegúrate de que los archivos `dataset_ingredientes.csv` y `nutricion/datos_nutricionales.csv` existen en las ubicaciones correctas
- El archivo `dataset_ingredientes.csv` debe contener las columnas 'plato' e 'ingredientes'
- El archivo `nutricion/datos_nutricionales.csv` debe contener las columnas 'plato', 'calorias', 'proteina', 'carbohidratos', 'grasa'
- El módulo `cargar_datos.py` ha sido actualizado para manejar rutas relativas y mostrar mensajes de error más claros

Si tienes problemas con importaciones de TensorFlow:
- Instala TensorFlow ejecutando: `pip install tensorflow`
- Si no puedes instalar TensorFlow, los scripts ahora usan manejo de errores para permitir ejecución parcial con funcionalidad limitada
- Los notebooks de entrenamiento (`nutricion/modelo_nutricion.ipynb`, `ingredientes_vision/modelo_vision_ingredientes.ipynb`) requieren TensorFlow para funcionar completamente
- Algunas funciones avanzadas no estarán disponibles si TensorFlow no está instalado
- Las importaciones de tensorflow.keras se manejan con manejo de errores en los archivos Python principales
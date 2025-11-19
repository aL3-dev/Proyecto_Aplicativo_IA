"""
Aplicación integral de identificación de platos bolivianos
Funcionalidad: Tomar una foto -> Identificar plato -> Determinar ingredientes -> Calcular valor nutricional
"""
import sys
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import json

# Importar TensorFlow con manejo de errores para diferentes versiones
TENSORFLOW_AVAILABLE = False
try:
    # Intentar con la estructura tradicional de tensorflow.keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        # Intentar con la nueva estructura (TF 2.16+) donde keras es independiente
        import tensorflow as tf
        from keras.models import load_model
        from keras.preprocessing.image import img_to_array, load_img
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        print("Advertencia: No se pudieron importar las funciones de Keras. Algunas funcionalidades pueden estar limitadas.")
        # Definir valores por defecto
        load_model = None
        img_to_array = None
        load_img = None

# Agregar directorios al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.cargar_datos import cargar_datos_ingredientes, cargar_datos_nutricion, procesar_ingredientes, combinar_datos
from utils.preprocesamiento import crear_etiquetas_nutricionales


class IdentificadorPlatosBolivianos:
    """
    Clase que integra todos los componentes del sistema para identificar platos bolivianos
    a partir de una imagen, determinar sus ingredientes y calcular su valor nutricional.
    """
    
    def __init__(self):
        self.modelo_vision = None
        self.modelo_nutricion = None
        self.mlb = None
        self.df_datos = None
        self.cargar_datos_referencia()
        
    def cargar_datos_referencia(self):
        """Carga los datos de referencia combinados"""
        df_ingredientes = cargar_datos_ingredientes('dataset_ingredientes.csv')
        df_nutricion = cargar_datos_nutricion('nutricion/datos_nutricionales.csv')
        self.df_datos = combinar_datos(df_ingredientes, df_nutricion)
        self.df_datos = crear_etiquetas_nutricionales(self.df_datos)
    
    def predecir_ingredientes_desde_imagen(self, imagen_path, threshold=0.5):
        """
        Predice los ingredientes presentes en una imagen de plato boliviano
        
        Args:
            imagen_path (str): Ruta a la imagen
            threshold (float): Umbral para considerar un ingrediente como presente
            
        Returns:
            list: Lista de ingredientes predichos
        """
        # Este método usaría el modelo de visión por computadora entrenado
        # Por ahora, simulamos la funcionalidad hasta que el modelo esté entrenado
        
        print(f"Analizando imagen: {imagen_path}")
        
        # Simulación de predicción de ingredientes basada en similitud visual
        # En la implementación real, aquí se usaría el modelo entrenado
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen {imagen_path}")
            return []
        
        # Simular predicción (esto se reemplazaría con el modelo real de visión)
        # Por ahora, usamos una lógica simple basada en el nombre del archivo
        nombre_archivo = os.path.basename(imagen_path).lower()
        
        # Detección básica basada en nombre del archivo
        ingredientes_simulados = []
        if 'papa' in nombre_archivo or 'patata' in nombre_archivo:
            ingredientes_simulados.append('papa')
        if 'carne' in nombre_archivo:
            ingredientes_simulados.append('carne de res')
        if 'pollo' in nombre_archivo:
            ingredientes_simulados.append('pollo')
        if 'cerdo' in nombre_archivo:
            ingredientes_simulados.append('carne de cerdo')
        if 'bagre' in nombre_archivo or 'pescado' in nombre_archivo:
            ingredientes_simulados.append('bagre')
        if 'cebolla' in nombre_archivo:
            ingredientes_simulados.append('cebolla')
        if 'tomate' in nombre_archivo:
            ingredientes_simulados.append('tomate')
        if 'ajo' in nombre_archivo:
            ingredientes_simulados.append('ajo')
        if 'maiz' in nombre_archivo or 'maíz' in nombre_archivo:
            ingredientes_simulados.append('maíz pelado')
        if 'chuño' in nombre_archivo:
            ingredientes_simulados.append('chuño')
        if 'arroz' in nombre_archivo:
            ingredientes_simulados.append('arroz')
        
        if not ingredientes_simulados:
            # Si no se identificaron ingredientes por nombre, usar algunos aleatorios
            # basados en tipos comunes de platos bolivianos
            ingredientes_posibles = ['cebolla', 'tomate', 'ajo', 'papa', 'carne de res', 
                                   'pollo', 'arroz', 'maíz pelado']
            import random
            ingredientes_simulados = random.sample(ingredientes_posibles, 
                                                  min(3, len(ingredientes_posibles)))
        
        return ingredientes_simulados
    
    def identificar_plato_mas_probable(self, ingredientes_predichos):
        """
        Identifica el plato más probable basado en los ingredientes encontrados
        
        Args:
            ingredientes_predichos (list): Lista de ingredientes predichos
            
        Returns:
            str: Nombre del plato más probable
            dict: Información nutricional del plato
        """
        if not ingredientes_predichos:
            return "No se pudo identificar el plato", {}
        
        # Calcular similitud entre ingredientes predichos y ingredientes reales
        mejor_coincidencia = ""
        mejor_puntaje = 0
        mejor_info_nutricional = {}
        
        for idx, row in self.df_datos.iterrows():
            ingredientes_reales = set(row['lista_ingredientes'])
            ingredientes_predichos_set = set(ingredientes_predichos)
            
            # Calcular intersección de ingredientes
            interseccion = len(ingredientes_reales.intersection(ingredientes_predichos_set))
            union = len(ingredientes_reales.union(ingredientes_predichos_set))
            
            if union > 0:
                similitud = interseccion / union
                if similitud > mejor_puntaje:
                    mejor_puntaje = similitud
                    mejor_coincidencia = row['plato']
                    mejor_info_nutricional = {
                        'calorias': row['calorias'],
                        'proteina': row['proteina'],
                        'carbohidratos': row['carbohidratos'],
                        'grasa': row['grasa'],
                        'categoria_proteina': row['categoria_proteina'],
                        'categoria_carbohidratos': row['categoria_carbohidratos'],
                        'categoria_grasa': row['categoria_grasa'],
                        'categoria_calorias': row['categoria_calorias']
                    }
        
        return mejor_coincidencia, mejor_info_nutricional

    def procesar_imagen_plato(self, imagen_path):
        """
        Procesa una imagen de plato y devuelve información completa
        
        Args:
            imagen_path (str): Ruta a la imagen del plato
            
        Returns:
            dict: Información completa del plato identificado
        """
        # 1. Predecir ingredientes desde la imagen
        ingredientes = self.predecir_ingredientes_desde_imagen(imagen_path)
        
        # 2. Identificar el plato más probable
        plato_identificado, info_nutricional = self.identificar_plato_mas_probable(ingredientes)
        
        # 3. Crear resultado completo
        resultado = {
            'imagen_procesada': imagen_path,
            'plato_identificado': plato_identificado,
            'ingredientes_detectados': ingredientes,
            'informacion_nutricional': info_nutricional
        }
        
        return resultado
    
    def mostrar_resultado(self, resultado):
        """Muestra el resultado de manera amigable"""
        print(f"\n=== RESULTADO DE ANÁLISIS ===")
        print(f"Imagen analizada: {resultado['imagen_procesada']}")
        print(f"Plato identificado: {resultado['plato_identificado']}")
        print(f"Ingredientes detectados: {', '.join(resultado['ingredientes_detectados'])}")
        
        if resultado['informacion_nutricional']:
            info = resultado['informacion_nutricional']
            print(f"\nInformación nutricional estimada:")
            print(f"  - Calorías: {info['calorias']} kcal")
            print(f"  - Proteína: {info['proteina']}g ({info['categoria_proteina']})")
            print(f"  - Carbohidratos: {info['carbohidratos']}g ({info['categoria_carbohidratos']})")
            print(f"  - Grasa: {info['grasa']}g ({info['categoria_grasa']})")
        else:
            print("No se pudo obtener información nutricional para este plato.")


def main():
    print("Iniciando aplicación de identificación de platos bolivianos...")
    print("Sistema que identifica platos a partir de imágenes y proporciona información nutricional")
    
    # Crear instancia del identificador
    identificador = IdentificadorPlatosBolivianos()
    
    # Directorios de ejemplo
    directorios_ejemplo = ['dataset_imagenes/Aji_de_pataskha', 'dataset_imagenes/caldo_de_bagre', 
                          'dataset_imagenes/Cazuelas', 'dataset_imagenes/Chairo']
    
    # Procesar una imagen de ejemplo si existen directorios
    for directorio in directorios_ejemplo:
        if os.path.exists(directorio):
            archivos = [f for f in os.listdir(directorio) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if archivos:
                imagen_ejemplo = os.path.join(directorio, archivos[0])
                print(f"\nProcesando imagen de ejemplo: {imagen_ejemplo}")
                
                try:
                    resultado = identificador.procesar_imagen_plato(imagen_ejemplo)
                    identificador.mostrar_resultado(resultado)
                except Exception as e:
                    print(f"Error procesando imagen: {e}")
                break
    else:
        print("No se encontraron imágenes de ejemplo en los directorios esperados.")
    
    print("\n=== Instrucciones para usar la aplicación ===")
    print("1. Tomar una foto de un plato boliviano")
    print("2. Usar la función procesar_imagen_plato() con la ruta de la imagen")
    print("3. El sistema identificará el plato, sus ingredientes y valor nutricional")


if __name__ == "__main__":
    main()
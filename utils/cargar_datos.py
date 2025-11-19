import pandas as pd
import os
from typing import Dict, List, Tuple


def cargar_datos_ingredientes(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el dataset de ingredientes desde un archivo CSV

    Args:
        ruta_csv (str): Ruta al archivo CSV con los datos de ingredientes

    Returns:
        pd.DataFrame: DataFrame con las columnas 'plato' e 'ingredientes'
    """
    # Verificar si el archivo existe y construir ruta absoluta si es necesario
    if not os.path.exists(ruta_csv):
        # Intentar buscar el archivo desde la raíz del proyecto
        proyecto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_csv_abs = os.path.join(proyecto_dir, ruta_csv)
        if os.path.exists(ruta_csv_abs):
            ruta_csv = ruta_csv_abs
        else:
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Verificar que las columnas necesarias existen
    if 'plato' not in df.columns or 'ingredientes' not in df.columns:
        raise ValueError(f"El archivo {ruta_csv} debe contener las columnas 'plato' e 'ingredientes'")

    return df


def cargar_datos_nutricion(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el dataset de información nutricional desde un archivo CSV

    Args:
        ruta_csv (str): Ruta al archivo CSV con los datos nutricionales

    Returns:
        pd.DataFrame: DataFrame con las columnas 'plato', 'calorias', 'proteina', 'carbohidratos', 'grasa'
    """
    # Verificar si el archivo existe y construir ruta absoluta si es necesario
    if not os.path.exists(ruta_csv):
        # Intentar buscar el archivo desde la raíz del proyecto
        proyecto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_csv_abs = os.path.join(proyecto_dir, ruta_csv)
        if os.path.exists(ruta_csv_abs):
            ruta_csv = ruta_csv_abs
        else:
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Verificar que las columnas necesarias existen
    columnas_necesarias = ['plato', 'calorias', 'proteina', 'carbohidratos', 'grasa']
    for col in columnas_necesarias:
        if col not in df.columns:
            raise ValueError(f"El archivo {ruta_csv} debe contener la columna '{col}'")

    return df


def procesar_ingredientes(ingredientes_str: str) -> List[str]:
    """
    Procesa la cadena de ingredientes y devuelve una lista

    Args:
        ingredientes_str (str): Cadena con ingredientes en formato JSON o similar

    Returns:
        List[str]: Lista de ingredientes
    """
    # Manejar valores nulos o vacíos
    if pd.isna(ingredientes_str) or ingredientes_str == '' or ingredientes_str == 'nan':
        return []

    # Convertir a string para asegurar el procesamiento
    ingredientes_str = str(ingredientes_str).strip()

    # Intentar evaluar como lista si comienza y termina con corchetes
    if ingredientes_str.startswith('[') and ingredientes_str.endswith(']'):
        try:
            # Intentar evaluar como una lista de Python
            import ast
            ingredientes_lista = ast.literal_eval(ingredientes_str)
            if isinstance(ingredientes_lista, list):
                return [str(ingrediente).strip().strip("'\"") for ingrediente in ingredientes_lista]
        except:
            # Si falla, procesar manualmente
            pass

    # Eliminamos corchetes y apóstrofos y convertimos a lista
    if ingredientes_str.startswith("['") and ingredientes_str.endswith("']"):
        ingredientes_str = ingredientes_str[2:-2]  # Removemos [' y ']
    elif ingredientes_str.startswith("'") and ingredientes_str.endswith("'"):
        ingredientes_str = ingredientes_str[1:-1]  # Removemos ' al inicio y final

    # Separar por comas y limpiar cada ingrediente
    ingredientes = [ing.strip().strip("'\"") for ing in ingredientes_str.split("','")]

    # Manejar el caso donde los ingredientes están entre [' y ']
    if len(ingredientes) == 1 and ',' in ingredientes_str:
        ingredientes = [ing.strip().strip("[]'\"") for ing in ingredientes_str.split(',')]

    # Eliminar entradas vacías
    ingredientes = [ing for ing in ingredientes if ing.strip()]

    return ingredientes


def combinar_datos(ingredientes_df: pd.DataFrame, nutricion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina los datos de ingredientes y nutrición en un solo DataFrame
    
    Args:
        ingredientes_df (pd.DataFrame): DataFrame con platos e ingredientes
        nutricion_df (pd.DataFrame): DataFrame con información nutricional
        
    Returns:
        pd.DataFrame: DataFrame combinado con todos los datos
    """
    # Aplicar la función para procesar los ingredientes en el dataframe
    ingredientes_df['lista_ingredientes'] = ingredientes_df['ingredientes'].apply(procesar_ingredientes)
    
    # Combinar los DataFrames usando la columna 'plato'
    df_combinado = pd.merge(ingredientes_df, nutricion_df, on='plato', how='inner')
    
    return df_combinado


def obtener_ruta_imagenes(base_path: str) -> Dict[str, List[str]]:
    """
    Obtiene las rutas de las imágenes de los platos
    
    Args:
        base_path (str): Ruta base del directorio de imágenes
        
    Returns:
        Dict[str, List[str]]: Diccionario con nombres de platos como clave y rutas de imágenes como valor
    """
    imagenes_dict = {}
    
    if os.path.exists(base_path):
        for plato_dir in os.listdir(base_path):
            plato_path = os.path.join(base_path, plato_dir)
            if os.path.isdir(plato_path):
                # Convertir nombre de directorio a formato consistente con el CSV
                nombre_plato = plato_dir.replace('_', ' ').title()
                # Algunos ajustes para coincidir con el formato del dataset
                if 'Orureno' in nombre_plato:
                    nombre_plato = nombre_plato.replace('Orureno', 'Orureño')
                if 'Beniano' in nombre_plato:
                    nombre_plato = nombre_plato.replace('Beniano', 'Beniano')
                
                imagenes = []
                for img_file in os.listdir(plato_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        imagenes.append(os.path.join(plato_path, img_file))
                
                if imagenes:
                    imagenes_dict[nombre_plato] = imagenes
                    
    return imagenes_dict
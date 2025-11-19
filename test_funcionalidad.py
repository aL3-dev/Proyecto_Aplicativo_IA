"""
Script de prueba para verificar la funcionalidad del proyecto
"""
import sys
import os

# Añadir el directorio utils al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.cargar_datos import cargar_datos_ingredientes, cargar_datos_nutricion, procesar_ingredientes, combinar_datos, obtener_ruta_imagenes
from utils.preprocesamiento import preprocesar_texto_ingredientes, crear_etiquetas_nutricionales


def test_carga_datos():
    """Prueba la carga de datos"""
    print("=== Prueba de Carga de Datos ===")

    try:
        # Cargar datasets
        df_ingredientes = cargar_datos_ingredientes('dataset_ingredientes.csv')
        df_nutricion = cargar_datos_nutricion('nutricion/datos_nutricionales.csv')

        print(f"[OK] Dataset de ingredientes cargado: {df_ingredientes.shape}")
        print(f"[OK] Dataset nutricional cargado: {df_nutricion.shape}")

        # Mostrar ejemplos
        print("\nEjemplo de ingredientes:")
        print(df_ingredientes.head(2))

        print("\nEjemplo de nutrición:")
        print(df_nutricion.head(2))

        return df_ingredientes, df_nutricion
    except Exception as e:
        print(f"[ERROR] Error en la carga de datos: {e}")
        return None, None


def test_procesamiento_ingredientes(df_ingredientes):
    """Prueba el procesamiento de ingredientes"""
    print("\n=== Prueba de Procesamiento de Ingredientes ===")

    try:
        # Probar procesamiento de ingredientes
        ejemplo_ingrediente = df_ingredientes['ingredientes'].iloc[0]
        print(f"Ingrediente original: {ejemplo_ingrediente}")

        ingredientes_procesados = procesar_ingredientes(ejemplo_ingrediente)
        print(f"Ingrediente procesado: {ingredientes_procesados}")

        # Combinar datasets
        df_combinado = combinar_datos(df_ingredientes, cargar_datos_nutricion('nutricion/datos_nutricionales.csv'))
        print(f"[OK] Dataset combinado: {df_combinado.shape}")

        print("\nEjemplo de combinación:")
        print(df_combinado[['plato', 'ingredientes', 'calorias', 'proteina']].head(2))

        return df_combinado
    except Exception as e:
        print(f"[ERROR] Error en el procesamiento de ingredientes: {e}")
        return None


def test_preprocesamiento(df_combinado):
    """Prueba el preprocesamiento"""
    print("\n=== Prueba de Preprocesamiento ===")

    try:
        df_procesado = preprocesar_texto_ingredientes(df_combinado)
        df_con_etiquetas = crear_etiquetas_nutricionales(df_procesado)

        print(f"[OK] Datos procesados y con etiquetas: {df_con_etiquetas.shape}")

        print("\nEjemplo de datos con categorías nutricionales:")
        print(df_con_etiquetas[['plato', 'categoria_proteina', 'categoria_carbohidratos', 'categoria_grasa', 'categoria_calorias']].head())

        return df_con_etiquetas
    except Exception as e:
        print(f"[ERROR] Error en el preprocesamiento: {e}")
        return None


def test_rutas_imagenes():
    """Prueba la obtención de rutas de imágenes"""
    print("\n=== Prueba de Rutas de Imágenes ===")

    try:
        imagenes_dict = obtener_ruta_imagenes('dataset_imagenes/')
        print(f"[OK] Directorios con imágenes encontrados: {len(imagenes_dict)}")

        # Mostrar algunos ejemplos
        for plato, rutas in list(imagenes_dict.items())[:3]:
            print(f"  {plato}: {len(rutas)} imágenes")
            if rutas:
                print(f"    Ejemplo: {rutas[0]}")

        return imagenes_dict
    except Exception as e:
        print(f"[ERROR] Error al obtener rutas de imágenes: {e}")
        return None


def main():
    print("Iniciando pruebas del proyecto de identificacion nutricional e ingredientes de platos bolivianos...\n")

    # Prueba de carga de datos
    df_ingredientes, df_nutricion = test_carga_datos()
    if df_ingredientes is None or df_nutricion is None:
        print("\n[ERROR] Las pruebas fallaron debido a errores en la carga de datos.")
        return

    # Prueba de procesamiento de ingredientes
    df_combinado = test_procesamiento_ingredientes(df_ingredientes)
    if df_combinado is None:
        print("\n[ERROR] Las pruebas fallaron debido a errores en el procesamiento de ingredientes.")
        return

    # Prueba de preprocesamiento
    df_procesado = test_preprocesamiento(df_combinado)
    if df_procesado is None:
        print("\n[ERROR] Las pruebas fallaron debido a errores en el preprocesamiento.")
        return

    # Prueba de rutas de imágenes
    imagenes_dict = test_rutas_imagenes()
    if imagenes_dict is None:
        print("\n[ERROR] Las pruebas fallaron debido a errores en la obtencion de rutas de imagenes.")
        return

    print("\n[SUCCESS] Todas las pruebas se completaron exitosamente!")


if __name__ == "__main__":
    main()
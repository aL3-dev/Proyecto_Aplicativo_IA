import os
import pandas as pd

# Leer los CSV
ingredientes_df = pd.read_csv('dataset_ingredientes.csv')
nutricion_df = pd.read_csv('nutricion/datos_nutricionales.csv')

# Obtener nombres de platos en CSV
platos_ingredientes = list(ingredientes_df['plato'])
platos_nutricion = list(nutricion_df['plato'])

# Obtener nombres de carpetas en directorio de imágenes
imagenes_dir = 'dataset_imagenes'
carpetas_imagenes = os.listdir(imagenes_dir)

print("=== ANALISIS DE CONCORDANCIA DE NOMBRES ===")
print()

print("Platos en ingredientes.csv:")
for plato in platos_ingredientes:
    print(f"  - '{plato}'")
print(f"Total: {len(platos_ingredientes)} platos")
print()

print("Platos en datos_nutricionales.csv:")
for plato in platos_nutricion:
    print(f"  - '{plato}'")
print(f"Total: {len(platos_nutricion)} platos")
print()

print("Carpetas en dataset_imagenes:")
for carpeta in carpetas_imagenes:
    print(f"  - '{carpeta}'")
print(f"Total: {len(carpetas_imagenes)} carpetas")
print()

# Función para normalizar nombres
def normalizar_nombre(nombre):
    # Reemplazar caracteres especiales y espacios
    nombre = nombre.lower()
    nombre = nombre.replace(" ", "_").replace("-", "_")
    nombre = nombre.replace("í", "i").replace("ó", "o").replace("á", "a").replace("é", "e").replace("ú", "u")
    nombre = nombre.replace("ñ", "n")
    return nombre

# Normalizar todos los nombres
platos_ingredientes_norm = [normalizar_nombre(p) for p in platos_ingredientes]
platos_nutricion_norm = [normalizar_nombre(p) for p in platos_nutricion]
carpetas_norm = [normalizar_nombre(c) for c in carpetas_imagenes]

print("=== ANALISIS DE CONCORDANCIA (NOMBRES NORMALIZADOS) ===")
print()

# Verificar concordancia entre ingredientes y carpetas
platos_ing_no_encontrados = []
for original, normalizado in zip(platos_ingredientes, platos_ingredientes_norm):
    if normalizado not in carpetas_norm:
        platos_ing_no_encontrados.append((original, normalizado))

print("Platos en ingredientes.csv sin carpeta correspondiente:")
for original, normalizado in platos_ing_no_encontrados:
    print(f"  - '{original}' -> '{normalizado}' (no encontrada en carpetas)")
print()

# Verificar concordancia entre nutricion y carpetas
platos_nut_no_encontrados = []
for original, normalizado in zip(platos_nutricion, platos_nutricion_norm):
    if normalizado not in carpetas_norm:
        platos_nut_no_encontrados.append((original, normalizado))

print("Platos en datos_nutricionales.csv sin carpeta correspondiente:")
for original, normalizado in platos_nut_no_encontrados:
    print(f"  - '{original}' -> '{normalizado}' (no encontrada en carpetas)")
print()

# Verificar carpetas sin datos en CSV
carpetas_sin_datos = []
for original, normalizado in zip(carpetas_imagenes, carpetas_norm):
    if normalizado not in platos_ingredientes_norm and normalizado not in platos_nutricion_norm:
        carpetas_sin_datos.append((original, normalizado))

print("Carpetas en dataset_imagenes sin datos en CSV:")
for original, normalizado in carpetas_sin_datos:
    print(f"  - '{original}' -> '{normalizado}' (no encontrada en CSVs)")
print()

print("=== RESUMEN ===")
print(f"Coincidencias entre ingredientes y carpetas: {len(platos_ingredientes) - len(platos_ing_no_encontrados)}/{len(platos_ingredientes)}")
print(f"Coincidencias entre nutricion y carpetas: {len(platos_nutricion) - len(platos_nut_no_encontrados)}/{len(platos_nutricion)}")
print(f"Carpetas sin datos en CSV: {len(carpetas_sin_datos)}/{len(carpetas_imagenes)}")
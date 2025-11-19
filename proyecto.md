# ÍNDICE
1. **Resumen**  
2. **Capítulo 1: Generalidades**  
   2.1 Introducción  
   2.2 Análisis de la problemática  
   2.3 Objetivos  
   - Objetivo general  
   - Objetivos específicos  
   2.4 Justificación  
   2.5 Alcances y limitaciones  

---

# RESUMEN

El presente proyecto propone el desarrollo de un modelo de aprendizaje supervisado utilizando un dataset de platos tradicionales bolivianos que incluye imágenes, descripciones e ingredientes. Se plantean dos posibles aplicaciones: (1) un sistema de **clasificación nutricional de platos bolivianos** y (2) un **modelo capaz de identificar ingredientes desde imágenes** mediante visión por computadora. Estas soluciones buscan contribuir al análisis nutricional accesible, la preservación gastronómica y la innovación tecnológica dentro del contexto boliviano. El proyecto comprende la preparación del dataset, entrenamiento de modelos de redes neuronales profundas y la evaluación de su desempeño mediante métricas comunes de machine learning.

---

# Capítulo 1: Generalidades

## 1.1 Introducción

En Bolivia, la gastronomía posee un rol cultural y social fundamental, con platos tradicionales que varían según región, ingredientes y técnicas culinarias. Sin embargo, existe una limitada disponibilidad de herramientas tecnológicas que permitan analizar la composición nutricional de estos platos o identificar sus ingredientes automáticamente a partir de imágenes.

El crecimiento del machine learning y la visión por computadora abre la posibilidad de desarrollar modelos inteligentes capaces de interpretar datos gastronómicos de forma automática. En este proyecto se utiliza un dataset de imágenes y descripciones de comida boliviana para construir dos tipos de sistemas basados en aprendizaje supervisado:  
1. Un **clasificador nutricional** que predice categorías como “alto en carbohidratos”, “alto en proteínas”, “alto en grasas”, etc.  
2. Un **modelo de identificación de ingredientes** que analiza una imagen y estima qué componentes contiene el plato.

Estos sistemas pueden ser aplicados en salud, gastronomía, educación alimentaria y aplicaciones móviles.

---

## 1.2 Análisis de la problemática

Actualmente, Bolivia carece de sistemas automatizados que permitan analizar nutricionalmente los platos típicos o identificar sus ingredientes mediante tecnología avanzada. Esto genera varios problemas:

- No existen herramientas digitales que faciliten a la población conocer el valor nutricional aproximado de los alimentos consumidos.  
- Nutricionistas, estudiantes y profesionales deben recurrir a métodos manuales para estimar ingredientes o clasificar comidas.  
- La preservación y sistematización de la gastronomía boliviana aún es limitada en el ámbito tecnológico.  
- Las aplicaciones internacionales de food recognition no contienen platos bolivianos, lo que limita su uso en el país.

El uso de un dataset especializado en comida boliviana permitiría entrenar modelos más precisos, relevantes y adaptados al contexto nacional.

---

## 1.3 Objetivos

### Objetivo general
Desarrollar un modelo de aprendizaje supervisado utilizando un dataset de platos tradicionales bolivianos, enfocado en la **clasificación nutricional** y/o **identificación automática de ingredientes desde imágenes**.

### Objetivos específicos
- Preprocesar el dataset de imágenes, descripciones e ingredientes de comida boliviana.  
- Entrenar un modelo de visión por computadora basado en redes neuronales convolucionales (CNN).  
- Implementar una arquitectura adecuada para clasificación de ingredientes o categorías nutricionales.  
- Evaluar el rendimiento del modelo mediante métricas como accuracy, precision, recall y F1-score.  
- Documentar el diseño, experimentación y resultados obtenidos.

---

## 1.4 Justificación

Este proyecto es relevante por varias razones:

- **Aporte a la salud pública:** un sistema automatizado puede ayudar a la población a conocer el valor nutritivo de los platos bolivianos.  
- **Preservación gastronómica:** permite crear herramientas inteligentes centradas en comida local, no solo en datasets internacionales.  
- **Innovación tecnológica:** impulsa el uso de aprendizaje profundo dentro del contexto boliviano.  
- **Aplicaciones prácticas:** puede servir para apps móviles, software para nutricionistas o sistemas educativos.  
- **Creciente demanda profesional:** la carrera de Ingeniería Informática requiere proyectos reales que apliquen IA en datos locales.

---

## 1.5 Alcances y limitaciones

### Alcances
- Se utilizará un dataset real de platos tradicionales bolivianos.  
- El modelo será entrenado para clasificación nutricional o identificación de ingredientes.  
- Se emplearán técnicas modernas de visión por computadora (CNN, transfer learning).  
- Se evaluará el desempeño del modelo con métricas estándar.  

### Limitaciones
- La precisión dependerá de la cantidad y calidad del dataset.  
- Ingredient recognition puede ser difícil en platos donde los ingredientes no son visualmente distinguibles.  
- La estimación nutricional puede requerir datos adicionales no incluidos en el dataset.  
- No se desarrollará una aplicación móvil completa; solo el modelo base.
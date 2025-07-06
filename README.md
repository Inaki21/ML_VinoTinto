# Predicción de Calidad de Vino

Este proyecto tiene como objetivo la clasificación y predicción de la calidad de vinos mediante técnicas de **Machine Learning**. El modelo utiliza **Random Forest** y **KMeans con PCA** para realizar tanto clasificación supervisada como no supervisada, ayudando a prever la calidad de un vino basándose en sus características químicas.

## **Estructura del Proyecto**

La estructura del proyecto es la siguiente:

|-- nombre_proyecto_final_ML
|-- data
| |-- raw
| |-- winequality-red.csv # Datos originales
| |-- processed
| |-- vinotinto.csv # Datos procesados y balanceados
|
|-- notebooks
| |-- 01_Fuentes_Limpieza.ipynb # Notebook de limpieza de datos
| |-- 02_Modelos.ipynb # Notebook de entrenamiento de modelos supervisados
| |-- 03_Modelo_NoSupervisado.ipynb # Notebook para el modelo de clustering (KMeans)
| |-- 04_SeleccionModelo.ipynb # Selección y comparación del mejor modelo
|
|-- src
| |-- limpieza.py # Script para limpiar y balancear los datos
| |-- evaluation.py # Script para evaluar los modelos
| |-- training.py # Script para entrenar el modelo
| |-- main.py # Script para realizar predicciones con el modelo
|
|-- models
| |-- random_forest_model.pkl # Modelo entrenado Random Forest
|
|-- docs
| |-- negocio.ppt # Presentación de negocio
| |-- ds.ppt # Presentación técnica de los resultados
| |-- memoria.md # Documentación del proyecto
|
|-- README.md


## **Descripción del Proyecto**

Este proyecto está diseñado para predecir la calidad de un vino utilizando un conjunto de datos que contiene diversas características químicas del vino, como la acidez, el pH, los sulfatos, entre otros. A partir de estos datos, el objetivo es clasificar el vino en tres categorías:

- **Bajo**
- **Medio**
- **Alto**

### **Metodología y Modelos Utilizados**

1. **Modelo Supervisado (Random Forest)**:
   - Se utiliza **Random Forest** para predecir la calidad del vino basado en sus características químicas.
   - El modelo es entrenado utilizando datos balanceados con **SMOTE**.
   - Se optimiza con **validación cruzada** y **GridSearchCV**.

2. **Modelo No Supervisado (KMeans con PCA)**:
   - Utilizamos **KMeans** para realizar un clustering y segmentación de los datos.
   - **PCA** se utiliza para reducir la dimensionalidad y visualizar los clusters.

### **Evaluación de los Modelos**
- **Random Forest** alcanzó un rendimiento promedio de **69%** de **accuracy** después de aplicar validación cruzada y ajustar los parámetros.
- **KMeans con PCA** mostró una **Silhouette Score** de **0.42**, lo que indica una segmentación moderada de los datos.

## **Instrucciones para ejecutar el proyecto**

1. **EJECUTAR limpieza.py**
2. **EJECUTAR evaluation.py**
3. **EJECUTAR training.py**
4. **EJECUTAR main.py**





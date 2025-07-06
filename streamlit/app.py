import streamlit as st
import pandas as pd
import pickle  # Si el modelo es de scikit-learn
import tensorflow as tf  # Si estás usando TensorFlow

# Cargar el modelo entrenado (asegúrate de que el archivo esté en el mismo directorio o ruta adecuada)
# Si el modelo es de scikit-learn, descomenta la siguiente línea:
# with open('modelo_vino.pkl', 'rb') as f:
#     modelo = pickle.load(f)

# Si el modelo es de TensorFlow, descomenta la siguiente línea:
#modelo = tf.keras.models.load_model('modelo_vino.h5')  # Cambia 'modelo_vino.h5' al nombre de tu archivo de modelo

# Título de la app
st.title("Predicción de Calidad del Vino")

# Ingreso de características del vino
st.header("Características del Vino")

# Campos para ingresar las características
fixed_acidity = st.number_input('Acidez fija', min_value=0.0, value=7.4)
volatile_acidity = st.number_input('Acidez volátil', min_value=0.0, value=0.7)
citric_acid = st.number_input('Ácido cítrico', min_value=0.0, value=0.0)
chlorides = st.number_input('Cloruros', min_value=0.0, value=0.076)
total_sulfur_dioxide = st.number_input('Dióxido de azufre total', min_value=0, value=34)
density = st.number_input('Densidad', min_value=0.0, value=0.9978)
sulphates = st.number_input('Sulfatos', min_value=0.0, value=0.56)
alcohol = st.number_input('Alcohol', min_value=0.0, value=9.4)

# Crear un DataFrame con las entradas
entrada = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'chlorides': [chlorides],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# Botón para realizar la predicción
# if st.button('Predecir Calidad del Vino'):
#     # Realizar la predicción utilizando el modelo cargado
#     prediccion = modelo.predict(entrada)
    
#     # Mapear el valor de la predicción a la categoría correspondiente
#     calidad = 'Malo' if prediccion == 0 else 'Bueno' if prediccion == 2 else 'Medio'
    
#     # Mostrar el resultado de la predicción
#     st.subheader(f"La predicción es: {calidad}")

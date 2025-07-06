import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def predecir(model_path, input_data_path):
    # Cargar el modelo entrenado
    model = joblib.load(model_path)

    # Cargar el conjunto de datos nuevo
    df = pd.read_csv(input_data_path)

    # Preprocesar los datos (escalar características)
    X = df.drop(['quality', 'quality_label', 'quality_label_encoded'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Realizar predicciones
    y_pred = model.predict(X_scaled)
    df['Predicción'] = y_pred

    # Mostrar las predicciones
    print("Predicciones realizadas:")
    print(df[['quality_label', 'Predicción']])

    # Guardar el archivo con las predicciones
    df.to_csv('predicciones.csv', index=False)
    print("Predicciones guardadas en 'predicciones.csv'")

if __name__ == "__main__":
    predecir('random_forest_model.pkl', 'data/new_data.csv')

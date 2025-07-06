import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def evaluar_modelo(model_path, X_test, y_test):
    # Cargar el modelo previamente entrenado
    model = joblib.load(model_path)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Imprimir el reporte de clasificación
    print("Evaluación del Modelo:")
    print(classification_report(y_test, y_pred))

    # Mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

if __name__ == "__main__":
    # Aquí se debe cargar el conjunto de prueba (X_test, y_test)
    # Y luego cargar y evaluar los modelos previamente entrenados
    X_test = ...  # Cargar el conjunto de prueba
    y_test = ...  # Cargar el conjunto de etiquetas
    evaluar_modelo('random_forest_model.pkl', X_test, y_test)

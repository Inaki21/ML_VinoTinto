import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

def entrenar_modelo(input_csv_path, model_output_path):
    # Leer los datos procesados
    df = pd.read_csv(input_csv_path)

    # Definir características (X) y objetivo (y)
    X = df.drop(['quality', 'quality_label', 'quality_label_encoded'], axis=1)
    y = df['quality_label_encoded']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inicializar y entrenar el modelo Random Forest
    rf = RandomForestClassifier(n_estimators=1000, max_depth=20, class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Validación cruzada
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Validación Cruzada (Accuracy): {cv_scores}")
    print(f"Promedio de Accuracy: {cv_scores.mean()}")

    # Guardar el modelo entrenado
    joblib.dump(rf, model_output_path)
    print(f"Modelo entrenado guardado en: {model_output_path}")

if __name__ == "__main__":
    entrenar_modelo('data/processed/vinotinto.csv', 'random_forest_model.pkl')

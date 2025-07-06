import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def limpiar_csv(input_csv_path, output_csv_path):
    # Leer el archivo CSV
    df = pd.read_csv(input_csv_path)

    # Realizar la limpieza de datos aquí (ej. eliminar nulos, etc.)
    df = df.dropna()

    # Convertir 'quality_label' en valores numéricos (si no está hecho)
    label_encoder = LabelEncoder()
    df['quality_label_encoded'] = label_encoder.fit_transform(df['quality_label'])

    # Definir características (X) y objetivo (y)
    X = df.drop(['quality', 'quality_label', 'quality_label_encoded'], axis=1)
    y = df['quality_label_encoded']

    # Balanceo de clases con SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Escalar las características
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    # Guardar el CSV procesado
    df_processed = pd.DataFrame(X_res_scaled, columns=X.columns)
    df_processed['quality_label_encoded'] = y_res
    df_processed.to_csv(output_csv_path, index=False)
    print(f"CSV procesado guardado en: {output_csv_path}")

if __name__ == "__main__":
    limpiar_csv('data/raw/winequality-red.csv', 'data/processed/vinotinto.csv')

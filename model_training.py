# model_training.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

def train_model(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=[
        "id_escola", "alta_evasao", "id_escola_nome", "id_municipio_nome",
        "sigla_uf_nome", "id_municipio", "inse_classificacao_2014", "inse_classificacao_2015"
    ])
    y = df["alta_evasao"]

    X = pd.get_dummies(X, columns=["sigla_uf", "rede"], drop_first=False)

    # Stratificar para manter proporção
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Imputação de valores ausentes com média
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())

    # Verificar o balanceamento das classes
    print(df["alta_evasao"].value_counts(normalize=True) * 100)
    print("Tipos de dados em X_train:\n", X_train.dtypes)
    print("Valores únicos em colunas problemáticas:")
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            print(f"{col}: {X_train[col].unique()}")

    # Treinamento do modelo (sem SMOTE)
    print("Treinando o modelo...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Modelo treinado.")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    model_path = "C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/evasion_model.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo salvo em {model_path}")

    columns_path = "C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/feature_columns.pkl"
    joblib.dump(X.columns.tolist(), columns_path)
    print(f"Colunas salvas em {columns_path}")

    return model, X.columns


if __name__ == "__main__":
    processed_data_path = "C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/processed_data.csv"
    train_model(processed_data_path)

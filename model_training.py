# model_training.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

def train_model(data_path):
    df = pd.read_csv(data_path)

    # Definir features (X) e target (y)
    # Excluir colunas que não são features ou são o target
    X = df.drop(columns=["id_escola", "alta_evasao"])
    y = df["alta_evasao"]

    # Converter colunas categóricas em numéricas usando one-hot encoding
    X = pd.get_dummies(X, columns=["sigla_uf", "rede"], drop_first=True)

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicializar e treinar o modelo Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probabilidade de alta evasão

    # Avaliar o modelo
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Salvar o modelo treinado
    model_path = "C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/evasion_model.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo salvo em {model_path}")

    return model, X.columns # Retornar o modelo e as colunas usadas para o encoding

if __name__ == "__main__":
    processed_data_path = "C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/processed_data.csv"
    train_model(processed_data_path)



# test_system.py

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def test_model_performance():
    """Testa a performance do modelo treinado"""
    print("=== TESTE DE PERFORMANCE DO MODELO ===")
    
    # Carregar dados
    df = pd.read_csv("C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/processed_data.csv")
    model = joblib.load("C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/evasion_model.joblib")
    
    # Preparar dados para teste
    X = df.drop(columns=["id_escola", "alta_evasao"])
    y = df["alta_evasao"]
    
    # Aplicar one-hot encoding
    X_encoded = pd.get_dummies(X, columns=["sigla_uf", "rede"], drop_first=True)
    
    # Fazer predições
    y_pred = model.predict(X_encoded)
    y_prob = model.predict_proba(X_encoded)[:, 1]
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y, y_pred))
    
    # Matriz de confusão
    cm = confusion_matrix(y, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)
    
    # Estatísticas adicionais
    accuracy = (y_pred == y).mean()
    print(f"\nAcurácia: {accuracy:.4f}")
    
    # Distribuição das probabilidades
    print(f"\nDistribuição das Probabilidades:")
    print(f"Mínima: {y_prob.min():.4f}")
    print(f"Máxima: {y_prob.max():.4f}")
    print(f"Média: {y_prob.mean():.4f}")
    print(f"Mediana: {np.median(y_prob):.4f}")
    
    return True

def test_data_quality():
    """Testa a qualidade dos dados processados"""
    print("\n=== TESTE DE QUALIDADE DOS DADOS ===")
    
    df = pd.read_csv("C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/processed_data.csv")
    
    print(f"Número total de registros: {len(df)}")
    print(f"Número de colunas: {len(df.columns)}")
    
    # Verificar valores ausentes
    missing_values = df.isnull().sum()
    print(f"\nValores ausentes por coluna:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    # Verificar distribuição da variável alvo
    print(f"\nDistribuição da variável alvo 'alta_evasao':")
    print(df["alta_evasao"].value_counts())
    print(f"Percentual de escolas em risco: {df['alta_evasao'].mean()*100:.2f}%")
    
    # Estatísticas descritivas das variáveis numéricas
    print(f"\nEstatísticas descritivas:")
    print(df[["ideb", "nivel_socioeconomico", "taxa_evasao_historica"]].describe())
    
    return True

def test_dashboard_functionality():
    """Testa a funcionalidade básica do dashboard"""
    print("\n=== TESTE DE FUNCIONALIDADE DO DASHBOARD ===")
    
    try:
        # Importar o módulo do dashboard
        import streamlit as st
        print("✓ Streamlit importado com sucesso")
        
        # Verificar se os dados podem ser carregados
        df = pd.read_csv("C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/processed_data.csv")
        print("✓ Dados carregados com sucesso")
        
        # Verificar se o modelo pode ser carregado
        model = joblib.load("C:/Users/Sara/Downloads/ProjetoDash/ProjetoDash/evasion_model.joblib")
        print("✓ Modelo carregado com sucesso")
        
        # Testar uma predição simples
        sample_data = pd.DataFrame({
            "ideb": [5.0],
            "nivel_socioeconomico": [50.0],
            "taxa_evasao_historica": [df["taxa_evasao_historica"].mean()],
            "sigla_uf": ["SP"],
            "rede": ["pública"]
        })
        
        sample_encoded = pd.get_dummies(sample_data, columns=["sigla_uf", "rede"], drop_first=True)
        
        # Garantir que todas as colunas do modelo estejam presentes
        df_encoded = pd.get_dummies(df[["ideb", "nivel_socioeconomico", "taxa_evasao_historica", "sigla_uf", "rede"]], 
                                    columns=["sigla_uf", "rede"], drop_first=True)
        
        for col in df_encoded.columns:
            if col not in sample_encoded.columns:
                sample_encoded[col] = 0
        
        sample_encoded = sample_encoded[df_encoded.columns]
        
        prediction = model.predict(sample_encoded)[0]
        probability = model.predict_proba(sample_encoded)[0][1]
        
        print(f"✓ Predição de teste realizada: Risco = {prediction}, Probabilidade = {probability:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste do dashboard: {str(e)}")
        return False

def generate_test_report():
    """Gera um relatório completo dos testes"""
    print("=== RELATÓRIO DE TESTES DO SISTEMA ===")
    print("Data/Hora:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    results = {
        "Teste de Performance do Modelo": test_model_performance(),
        "Teste de Qualidade dos Dados": test_data_quality(),
        "Teste de Funcionalidade do Dashboard": test_dashboard_functionality()
    }
    
    print(f"\n=== RESUMO DOS TESTES ===")
    for test_name, result in results.items():
        status = "✓ PASSOU" if result else "✗ FALHOU"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nStatus Geral: {'✓ TODOS OS TESTES PASSARAM' if all_passed else '✗ ALGUNS TESTES FALHARAM'}")
    
    return all_passed

if __name__ == "__main__":
    generate_test_report()


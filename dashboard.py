# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Preditor de Evasão Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎓 Preditor de Evasão Escolar no Ensino Médio")
st.markdown("---")

# Carregar dados e modelo
@st.cache_data
def load_data():
    df = pd.read_csv("/home/ubuntu/ProjetoDash/ProjetoDash/processed_data.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("/home/ubuntu/ProjetoDash/ProjetoDash/evasion_model.joblib")
    return model

# Carregar dados
df = load_data()
model = load_model()

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.selectbox("Escolha uma página:", 
                           ["Dashboard Principal", "Mapa de Risco", "Ranking de Fatores", 
                            "Comparativo Redes", "Simulador de Cenários"])

if page == "Dashboard Principal":
    st.header("📊 Dashboard Principal")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_escolas = len(df)
        st.metric("Total de Escolas", f"{total_escolas:,}")
    
    with col2:
        escolas_risco = df[df["alta_evasao"] == 1].shape[0]
        st.metric("Escolas em Risco", f"{escolas_risco:,}")
    
    with col3:
        taxa_media_evasao = df["taxa_evasao_historica"].mean()
        st.metric("Taxa Média de Evasão", f"{taxa_media_evasao:.2f}%")
    
    with col4:
        ideb_medio = df["ideb"].mean()
        st.metric("IDEB Médio", f"{ideb_medio:.2f}")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Estado")
        fig_uf = px.bar(df.groupby("sigla_uf")["alta_evasao"].sum().reset_index(),
                        x="sigla_uf", y="alta_evasao",
                        title="Escolas em Risco por Estado")
        st.plotly_chart(fig_uf, use_container_width=True)
    
    with col2:
        st.subheader("Distribuição por Rede")
        fig_rede = px.pie(df, names="rede", values="alta_evasao",
                          title="Distribuição de Risco por Rede")
        st.plotly_chart(fig_rede, use_container_width=True)

elif page == "Mapa de Risco":
    st.header("🗺️ Mapa de Escolas por Risco de Evasão")
    
    # Criar dados agregados por estado
    df_estado = df.groupby("sigla_uf").agg({
        "alta_evasao": "sum",
        "taxa_evasao_historica": "mean",
        "id_escola": "count"
    }).reset_index()
    df_estado.columns = ["Estado", "Escolas_Risco", "Taxa_Media_Evasao", "Total_Escolas"]
    df_estado["Percentual_Risco"] = (df_estado["Escolas_Risco"] / df_estado["Total_Escolas"]) * 100
    
    # Mapa coroplético
    fig_map = px.choropleth(
        df_estado,
        locations="Estado",
        color="Percentual_Risco",
        hover_name="Estado",
        hover_data=["Escolas_Risco", "Total_Escolas", "Taxa_Media_Evasao"],
        color_continuous_scale="Reds",
        title="Percentual de Escolas em Risco por Estado",
        locationmode="geojson-id"
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Tabela de dados
    st.subheader("Dados por Estado")
    st.dataframe(df_estado.sort_values("Percentual_Risco", ascending=False))

elif page == "Ranking de Fatores":
    st.header("📈 Ranking de Fatores Mais Influentes")
    
    # Análise de correlação
    correlations = df[["ideb", "nivel_socioeconomico", "taxa_evasao_historica"]].corr()["alta_evasao"].abs().sort_values(ascending=False)
    
    # Gráfico de barras
    fig_corr = px.bar(
        x=correlations.index,
        y=correlations.values,
        title="Correlação dos Fatores com Risco de Evasão",
        labels={"x": "Fatores", "y": "Correlação (Valor Absoluto)"}
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Análise detalhada
    st.subheader("Análise Detalhada dos Fatores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ideb = px.box(df, x="alta_evasao", y="ideb", 
                          title="Distribuição do IDEB por Risco de Evasão")
        st.plotly_chart(fig_ideb, use_container_width=True)
    
    with col2:
        fig_nse = px.box(df, x="alta_evasao", y="nivel_socioeconomico",
                         title="Distribuição do Nível Socioeconômico por Risco de Evasão")
        st.plotly_chart(fig_nse, use_container_width=True)

elif page == "Comparativo Redes":
    st.header("🏫 Comparativo entre Redes (Pública/Privada)")
    
    # Análise por rede
    df_rede = df.groupby("rede").agg({
        "alta_evasao": ["sum", "mean"],
        "taxa_evasao_historica": "mean",
        "ideb": "mean",
        "nivel_socioeconomico": "mean",
        "id_escola": "count"
    }).round(2)
    
    df_rede.columns = ["Escolas_Risco", "Percentual_Risco", "Taxa_Media_Evasao", 
                       "IDEB_Medio", "NSE_Medio", "Total_Escolas"]
    df_rede = df_rede.reset_index()
    
    # Gráficos comparativos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_comp1 = px.bar(df_rede, x="rede", y="Percentual_Risco",
                           title="Percentual de Risco por Rede")
        st.plotly_chart(fig_comp1, use_container_width=True)
    
    with col2:
        fig_comp2 = px.bar(df_rede, x="rede", y="IDEB_Medio",
                           title="IDEB Médio por Rede")
        st.plotly_chart(fig_comp2, use_container_width=True)
    
    # Tabela comparativa
    st.subheader("Tabela Comparativa")
    st.dataframe(df_rede)

elif page == "Simulador de Cenários":
    st.header("🎯 Simulador de Cenários")
    
    st.markdown("Use os controles abaixo para simular diferentes cenários e ver a predição de risco de evasão:")
    
    # Controles para simulação
    col1, col2 = st.columns(2)
    
    with col1:
        ideb_sim = st.slider("IDEB", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        nse_sim = st.slider("Nível Socioeconômico", min_value=30.0, max_value=80.0, value=50.0, step=1.0)
    
    with col2:
        uf_sim = st.selectbox("Estado", df["sigla_uf"].unique())
        rede_sim = st.selectbox("Rede", df["rede"].unique())
    
    # Preparar dados para predição
    # Criar um dataframe com os valores simulados
    sim_data = pd.DataFrame({
        "ideb": [ideb_sim],
        "nivel_socioeconomico": [nse_sim],
        "sigla_uf": [uf_sim],
        "rede": [rede_sim],
        "taxa_evasao_historica": [df["taxa_evasao_historica"].mean()]  # Adicionar a coluna que estava faltando
    })
    
    # Aplicar one-hot encoding igual ao usado no treinamento
    sim_encoded = pd.get_dummies(sim_data, columns=["sigla_uf", "rede"], drop_first=True)
    
    # Garantir que todas as colunas do modelo estejam presentes
    # Recriar as colunas do modelo
    df_encoded = pd.get_dummies(df[["ideb", "nivel_socioeconomico", "taxa_evasao_historica", "sigla_uf", "rede"]], 
                                columns=["sigla_uf", "rede"], drop_first=True)
    
    for col in df_encoded.columns:
        if col not in sim_encoded.columns:
            sim_encoded[col] = 0
    
    # Reordenar colunas para corresponder ao modelo
    sim_encoded = sim_encoded[df_encoded.columns]
    
    # Fazer predição
    try:
        probabilidade = model.predict_proba(sim_encoded)[0][1]
        risco = model.predict(sim_encoded)[0]
        
        # Mostrar resultado
        st.subheader("Resultado da Simulação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Probabilidade de Alta Evasão", f"{probabilidade:.2%}")
        
        with col2:
            status = "Alto Risco" if risco == 1 else "Baixo Risco"
            color = "red" if risco == 1 else "green"
            st.markdown(f"**Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        
        # Gráfico de probabilidade
        fig_prob = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probabilidade * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidade de Evasão (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        st.plotly_chart(fig_prob, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")

# Rodapé
st.markdown("---")
st.markdown("**Predição de Evasão Escolar** - Sistema desenvolvido para identificar escolas com risco crítico de evasão no ensino médio.")


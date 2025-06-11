# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from fpdf import FPDF
import plotly.io as pio
import os



def salvar_grafico(fig, filename):
    temp_path = os.path.join("/tmp", filename)
    pio.write_image(fig, temp_path, format='png', width=800, height=500)
    return temp_path

def gerar_pdf(df, fig1, fig2):
    salvar_grafico(fig1, "grafico1.png")
    salvar_grafico(fig2, "grafico2.png")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Evasão Escolar", ln=True, align="C")

    # Adicionar gráficos
    pdf.ln(10)
    pdf.image("grafico1.png", x=10, y=30, w=180)
    pdf.ln(80)
    pdf.image("grafico2.png", x=10, y=120, w=180)

    # Adicionar tabela com dados (exemplo limitado)
    pdf.ln(90)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Exemplo de Dados:", ln=True)
    for i, row in df.head(10).iterrows():
        linha = f"{row['sigla_uf']}, {row['rede']}, IDEB: {row['ideb']:.2f}, NSE: {row['nivel_socioeconomico']:.2f}"
        pdf.cell(200, 10, txt=linha, ln=True)

    pdf.output("relatorio_evasao.pdf")



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
    df = pd.read_csv("https://media.githubusercontent.com/media/BrendoGarcia/BigData/main/processed_data.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("evasion_model.joblib")
    columns = joblib.load("feature_columns.pkl")
    return model,columns
# Carregar dados
df = load_data()
model, load_columns = load_model()

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
        # Botão para gerar PDF
    st.markdown("### 📄 Gerar Relatório em PDF")

    if st.button("Gerar Relatório"):
        try:
            gerar_pdf(df, fig_uf, fig_rede)
            with open("relatorio_evasao.pdf", "rb") as f:
                st.download_button("📥 Baixar PDF", f, file_name="relatorio_evasao.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {str(e)}")
    

elif page == "Mapa de Risco":
    st.header("🗺️ Mapa de Escolas por Risco de Evasão")
    
    # Verificar se há colunas de coordenadas
    if "id_escola_latitude" in df.columns and "id_escola_longitude" in df.columns:
        # Filtrar apenas escolas com alta evasão
        df_risco = df[df["alta_evasao"] == 1]

        # Mapa de dispersão geográfica
        fig_map = px.scatter_mapbox(
            df_risco,
            lat="id_escola_latitude",
            lon="id_escola_longitude",
            color="taxa_evasao_historica",
            size="taxa_evasao_historica",
            hover_name="id_escola_nome" if "id_escola_nome" in df.columns else None,
            hover_data=["sigla_uf", "id_municipio_nome", "taxa_evasao_historica"],
            color_continuous_scale="Reds",
            size_max=15,
            zoom=3,
            height=600,
        )

        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("Escolas com Alta Evasão")
        st.dataframe(df_risco[["id_escola_nome", "sigla_uf", "id_municipio_nome", "taxa_evasao_historica"]])
    else:
        st.warning("Dados de latitude e longitude não encontrados.")


elif page == "Ranking de Fatores":
    st.header("📈 Ranking de Fatores Mais Influentes")
    
    # Análise de correlação
    correlations = df[["ideb","nota_saeb_media_padronizada","nivel_socioeconomico", "taxa_evasao_historica", "alta_evasao"]].corr()["alta_evasao"].abs().sort_values(ascending=False)
    
    # Gráfico de barras
    fig_corr = px.bar(
        x=correlations.index,
        y=correlations.values,
        title="Correlação dos Fatores com Risco de Evasão",
        labels={"x": "Fatores", "y": "Correlação (Valor Absoluto)"}
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    

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
        nse_sim = st.slider("Nível Socioeconômico", min_value=10.0, max_value=70.0, value=50.0, step=1.0)
    
    with col2:
        uf_sim = st.selectbox("Estado", df["sigla_uf"].unique())
        rede_sim = st.selectbox("Rede", df["rede"].unique())
    
    # Preparar dados para predição
    # Criar um dataframe com os valores simulados
    # Criar o dataframe com os valores simulados
    sim_data = pd.DataFrame({
        "ideb": [ideb_sim],
        "nivel_socioeconomico": [nse_sim],
        "sigla_uf": [uf_sim],
        "rede": [rede_sim],
        "taxa_evasao_historica": [df["taxa_evasao_historica"].mean()]
    })

    # Aplicar one-hot encoding
    sim_encoded = pd.get_dummies(sim_data, columns=["sigla_uf", "rede"])

    # Garantir que todas as colunas esperadas estejam presentes
    for col in load_columns:
        if col not in sim_encoded.columns:
            sim_encoded[col] = 0  # ou [0], ambos funcionam

    # Reordenar as colunas
    sim_encoded = sim_encoded[load_columns]

    
    # Fazer predição
    try:
        st.write("Dados simulados:", sim_data)
        st.write("Dados codificados:", sim_encoded)
        st.write("🔥 Colunas ativadas (dummies = 1):", sim_encoded.loc[:, sim_encoded.iloc[0] == 1])
        st.write("Proba completa:", model.predict_proba(sim_encoded))
        st.write("Classe predita:", model.predict(sim_encoded))
        st.write("Probabilidade (raw):", model.predict_proba(sim_encoded))


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

### Apartir daqui não tenho certeza mais de nada pode ser que funcione ou não.

# Rodapé
st.markdown("---")
st.markdown("**Predição de Evasão Escolar** - Sistema desenvolvido para identificar escolas com risco crítico de evasão no ensino médio.")


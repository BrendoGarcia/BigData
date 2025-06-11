# data_preparation.py

import pandas as pd

def load_data(base_path):
    ideb_path = f"{base_path}/DadosBrutos/ideb/bq-results-20250608-215951-1749420307594.csv"
    nse_path = f"{base_path}/DadosBrutos/nivel_socioeconomico/br_inep_indicadores_educacionais_escola_nivel_socioeconomico.csv"
    taxa_transicao_path = f"{base_path}/DadosBrutos/taxa_transicao/br_inep_indicadores_educacionais_uf_taxa_transicao.csv"

    df_ideb = pd.read_csv(ideb_path)
    df_nse = pd.read_csv(nse_path)
    df_taxa_transicao = pd.read_csv(taxa_transicao_path)

    return df_ideb, df_nse, df_taxa_transicao

def preprocess_data(df_ideb, df_nse, df_taxa_transicao):
    # Padronizar a coluna 'rede'
    def standardize_rede(df):
        df["rede"] = df["rede"].str.lower()
        df["rede"] = df["rede"].replace({
            "estadual": "pública",
            "municipal": "pública",
            "federal": "pública",
            "privada": "privada"
        })
        return df

    df_ideb = standardize_rede(df_ideb)
    df_nse = standardize_rede(df_nse)
    df_taxa_transicao = standardize_rede(df_taxa_transicao)

    # IDEB
    df_ideb_filtered = df_ideb.copy()
    df_ideb_filtered = df_ideb_filtered[[
    "id_escola", "id_escola_nome", "sigla_uf", "sigla_uf_nome",
    "id_municipio", "id_municipio_nome","rede",
    "id_escola_latitude", "id_escola_longitude", "ideb", "taxa_aprovacao", "indicador_rendimento",
    "nota_saeb_matematica", "nota_saeb_lingua_portuguesa", "nota_saeb_media_padronizada"]]

    # Nível Socioeconômico - Usando ano de 2015, pois é o mais recente disponível
    df_nse_filtered = df_nse.copy()
    df_nse_filtered = df_nse_filtered[["id_escola","inse_quantidade_alunos","valor_inse","inse_classificacao_2014","inse_classificacao_2015"]]
    df_nse_filtered.rename(columns={"valor_inse": "nivel_socioeconomico"}, inplace=True)

    # Taxa de Transição (Evasão Histórica) - Agregando por UF e rede
    df_taxa_transicao_filtered = df_taxa_transicao.copy()
    df_taxa_transicao_filtered = df_taxa_transicao_filtered[["sigla_uf", "rede", "taxa_evasao_em", "localizacao","taxa_promocao_em","taxa_promocao_em_1_ano","taxa_promocao_em_2_ano","taxa_promocao_em_3_ano","taxa_repetencia_em","taxa_repetencia_em_1_ano","taxa_repetencia_em_2_ano","taxa_repetencia_em_3_ano","taxa_evasao_em_1_ano","taxa_evasao_em_2_ano","taxa_evasao_em_3_ano"]]
    df_taxa_transicao_filtered.rename(columns={"taxa_evasao_em": "taxa_evasao_historica"}, inplace=True)
    df_taxa_transicao_filtered = df_taxa_transicao_filtered.groupby(["sigla_uf", "rede"])["taxa_evasao_historica"].mean().reset_index()

    # Debugging: Check distribution of 'taxa_evasao_historica' before merge
    print("\n--- Distribution of 'taxa_evasao_historica' in df_taxa_transicao_filtered ---")
    print(df_taxa_transicao_filtered["taxa_evasao_historica"].describe())

    # Unir os dataframes
    df_merged = pd.merge(df_ideb_filtered, df_nse_filtered, on="id_escola", how="left")
    df_merged = pd.merge(df_merged, df_taxa_transicao_filtered, on=["sigla_uf", "rede"], how="left")

    # Tratar valores ausentes
    df_merged["ideb"] = df_merged["ideb"].fillna(df_merged["ideb"].mean())
    df_merged["nivel_socioeconomico"] = df_merged["nivel_socioeconomico"].fillna(df_merged["nivel_socioeconomico"].mean())
    df_merged["taxa_evasao_historica"] = df_merged["taxa_evasao_historica"].fillna(df_merged["taxa_evasao_historica"].mean())

    # Criar a variável alvo: alta_evasao (>20%)
    # Ajustando o limiar para garantir a existência de ambas as classes
    # Se a taxa máxima for menor que 20, o limiar será ajustado para um valor que crie duas classes
    max_evasion = df_merged["taxa_evasao_historica"].max()
    threshold = 20
    if max_evasion <= threshold:
        # Se todas as taxas forem <= 20, use a mediana ou um percentil para criar classes
        # Para este exemplo, vamos usar a mediana como um limiar temporário se 20 não funcionar
        threshold = df_merged["taxa_evasao_historica"].median()
        print(f"Ajustando o limiar de evasão para {threshold:.2f} para criar classes balanceadas.")

    df_merged["alta_evasao"] = (df_merged["taxa_evasao_historica"] > threshold).astype(int)

    return df_merged

if __name__ == "__main__":
    base_path = "/BigData"
    df_ideb, df_nse, df_taxa_transicao = load_data(base_path)
    df_final = preprocess_data(df_ideb, df_nse, df_taxa_transicao)
    print(df_final.head())
    print(df_final.info())
    df_final.to_csv(f"{base_path}/processed_data.csv", index=False)
    print(f"Dados processados salvos em {base_path}/processed_data.csv")



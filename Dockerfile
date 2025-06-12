FROM python:3.10-slim

# Instalações básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório
WORKDIR /app

# Copia arquivos
COPY . /app

# Instala dependências
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expõe a porta (não obrigatório, mas ajuda)
EXPOSE 8080

# Comando para iniciar o app com porta dinâmica
CMD streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0

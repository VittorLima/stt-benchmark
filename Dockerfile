# Usa uma imagem base com CUDA e cuDNN pré-instalados
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Ajusta variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Sao_Paulo \
    PYTHONWARNINGS=ignore \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 

# Instala dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        ffmpeg \
        libsndfile1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/* /tmp/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de requisitos 
COPY requirements.txt .

# Instala as dependências do Python (incluindo PyTorch com suporte CUDA)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1+cu121 \
        torchaudio==2.5.1+cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip /tmp/*

# Copia o código fonte da aplicação
COPY src/ ./src/

# Define o comando de entrada para o container
CMD ["/bin/bash"]
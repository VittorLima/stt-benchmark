# Script para executar o container Docker

docker run \
    -it \
    --rm \
    --gpus all \
    --ipc=host \
    -v $(pwd):/app \
    --env-file .env \
    stt-benchmark-imagem:latest       # nome da imagem criada por você
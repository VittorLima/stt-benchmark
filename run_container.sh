# Script para executar o container Docker

docker run \
    -it \
    --rm \
    --gpus all \
    --ipc=host \
    -v $(pwd):/app \
    stt-benchmark-imagem:latest       # nome da imagem criada por você
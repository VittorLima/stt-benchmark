# stt-benchmark

Benchmark simples de modelos STT com WER, CER e tempo médio de inferência.

## 1) Dataset (essencial)

Coloque os arquivos nestes diretórios:

```text
dataset/
	audio/          # .wav, .mp3, .opus
	transcripts/    # .txt com o mesmo nome do áudio
```

Exemplo:

- `dataset/audio/audio_teste.wav`
- `dataset/transcripts/audio_teste.txt`

Regra simples: sem o `.txt` correspondente, o audio e ignorado no benchmark.

## 2) Rodar no container (.sh)

Na raiz do projeto:

Etapa 1: build da imagem

```bash
docker build -t stt-benchmark-imagem:latest .
```

Etapa 2: subir o container

```bash
./run_container.sh
```

## 3) Executar com argparse

Dentro do container:

```bash
python3 src/benchmark.py --model FasterWhisper
```

Troque `FasterWhisper` por qualquer modelo listado abaixo.

Modelos disponíveis hoje:

- `Deepgram`
- `Elevenlabs`
- `Parakeet`
- `FasterWhisper`
- `Speechmatics`

Se usar modelos de API, configure as chaves no `.env` (na raiz):

- `DEEPGRAM_API_KEY`
- `ELEVENLABS_API_KEY`
- `SPEECHMATICS_API_KEY`

## 4) Adicionar um novo modelo

1. Crie um arquivo em `src/models/` (ex.: `novo_modelo.py`).
2. Implemente uma classe com método `transcribe(audio_path: str) -> str`.
3. Registre o modelo em `src/models/__init__.py`:
   - adicione o nome em `MODELOS`
   - adicione o mapeamento no `_map`.
4. Rode:

```bash
python3 src/benchmark.py --model NomeDoModelo
```

## Estrutura de arquivos

```text
stt-benchmark/
├── dataset/                  
│   ├── audio/                # .wav, .mp3, .opus
│   └── transcripts/          # .txt com mesmo nome do áudio
├── src/                      
│   ├── benchmark.py          # executa benchmark e calcula métricas
│   ├── config.py             # config geral e variáveis de ambiente
│   └── models/               
│       ├── __init__.py       # registro dos modelos disponíveis
│       └── modelo_X.py       
├── Dockerfile                # definição da imagem Docker
├── run_container.sh          # comando pronto para rodar container
└── requirements.txt          # dependências Python
```
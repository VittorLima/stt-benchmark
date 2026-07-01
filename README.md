# stt-benchmark

Benchmark de modelos STT (Speech-to-Text) com WER, CER e tempo médio de inferência, usando o dataset CORAA-MUPE-ASR de fala espontânea em português brasileiro.

## Pré-requisitos

- Docker + NVIDIA Container Toolkit
- GPU NVIDIA
- Conexão à internet (para baixar o dataset do Hugging Face)

## Configuração

```bash
cp .env.example .env
```

Variáveis disponíveis (obrigatórias apenas para os modelos de API correspondentes):

| Variável | Padrão | Descrição |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Nível de log |
| `DEEPGRAM_API_KEY` | — | Chave da API do modelo `Deepgram` |
| `ELEVENLABS_API_KEY` | — | Chave da API do modelo `Elevenlabs` |
| `SPEECHMATICS_API_KEY` | — | Chave da API do modelo `Speechmatics` |

## Execução

```bash
docker build -t stt-benchmark-imagem:latest .
docker compose run --rm stt-benchmark
```

Dentro do container:

```bash
python3 src/benchmark.py --model FasterWhisper
```

Para limitar a quantidade de amostras processadas (e baixadas, se necessário), use `--samples`:

```bash
python3 src/benchmark.py --model FasterWhisper --samples 100
```

Exemplo de saída ao final da execução:

```text
INFO - Benchmark - Processados: 95/100 arquivos (5 pulados) — WER: 15.20%, CER: 8.45%, Tempo de inferência: 0.82s
```

O benchmark é robusto contra falhas: se uma transcrição falhar ou a referência estiver faltando/vazia, o arquivo é pulado e o processamento continua.

## Modelos disponíveis

| Modelo | Tipo | Requer API |
|---|---|---|
| `FasterWhisper` | Local | ✗ |
| `Whisper` | Local | ✗ |
| `Parakeet` | Local | ✗ |
| `Deepgram` | API | ✓ |
| `Elevenlabs` | API | ✓ |
| `Speechmatics` | API | ✓ |

## Medir latência (áudio único)

Mede o tempo médio de inferência de um modelo processando um único áudio de ~1 min (`samples/amostra.opus`), repetindo a transcrição 20x:

```bash
python3 src/latency.py --model FasterWhisper
```

Exemplo de saída:

```text
INFO - Latency - Processadas: 20/20 inferências (0 puladas) — Tempo médio de inferência: 0.75s
```

Para usar outro áudio, substitua o arquivo em `samples/amostra.opus` (ou ajuste `SAMPLE_AUDIO` em `src/latency.py`).

## Dataset

### CORAA-MUPE-ASR (padrão)

- **Total de amostras** (split de teste): ~30.968
- **Após filtros** (`audio_quality=high` + duração 5–20s): ~6.867 amostras
- **Formato**: arquivos em `dataset/audio/*.wav` e `dataset/transcripts/*.txt`

Comportamento do download:
- **Sem `--samples`**: baixa todas as ~6.867 amostras filtradas.
- **Com `--samples N`**: baixa apenas `N` amostras (se necessário); se já houver `≥ N` arquivos locais, pula o download.

### Dataset próprio

Salve pares `.wav` + `.txt` com o mesmo nome em `dataset/audio` e `dataset/transcripts`:

```
dataset/
├── audio/
│   ├── amostra_1.wav
│   └── ...
└── transcripts/
    ├── amostra_1.txt
    └── ...
```

Execute com `--samples` menor ou igual à quantidade de `.wav` locais para pular o download automático:

```bash
python3 src/benchmark.py --model FasterWhisper --samples 50
```

## Adicionar novo modelo

1. Crie um arquivo em `src/models/` (ex.: `novo_modelo.py`) com uma classe implementando `transcribe(audio_path: str) -> str`:

```python
class NomeDoModelo:
    def __init__(self) -> None:
        logger.info("Inicializando NomeDoModelo")

    def transcribe(self, audio_path: str) -> str:
        # Lógica de transcrição
        return "texto transcrito"
```

2. Registre o modelo em `src/models/__init__.py`: adicione o nome em `MODELOS` e o mapeamento em `_map`.
3. Execute:

```bash
python3 src/benchmark.py --model NomeDoModelo
```

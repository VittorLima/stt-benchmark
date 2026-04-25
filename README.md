# stt-benchmark

Benchmark simples de modelos STT com WER, CER e tempo médio de inferência.

## 1) Rodar no container

Na raiz do projeto:

Build da imagem:

```bash
docker build -t stt-benchmark-imagem:latest .
```

Subir o container:

```bash
./run_container.sh
```

## 2) Executar benchmark

Dentro do container:

```bash
python3 src/benchmark.py --model FasterWhisper
```

Para limitar a quantidade de amostras baixadas, use `--samples`:

```bash
python3 src/benchmark.py --model FasterWhisper --samples 100
```

Exemplo de saída ao final da execução:

```text
INFO - Benchmark - Resultados agregados - WER: 15.20%, CER: 8.45%, Tempo de inferência: 0.82s
```

## 3) Dataset

- O benchmark usa `dataset/audio/*.wav` e `dataset/transcripts/*.txt`.
- Sem `--samples`, o script baixa o dataset público automaticamente do Hugging Face (`nilc-nlp/CORAA-MUPE-ASR`, um corpus de português brasileiro) e salva em `dataset/`.
- Com `--samples N`, o download só ocorre quando há menos de `N` arquivos `.wav` locais.
- Filtros aplicados no download: `audio_quality=high` e duração entre 2 e 10 segundos.

Para usar dataset próprio:

- Salve pares `.wav` e `.txt` com o mesmo nome em `dataset/audio` e `dataset/transcripts`.
- Execute com `--samples N`, onde `N` seja menor ou igual à quantidade de `.wav` local (assim o download automático é pulado).

## 4) Modelos disponíveis

- `Deepgram`
- `Elevenlabs`
- `Parakeet`
- `FasterWhisper`
- `Speechmatics`

Se usar modelos de API, configure no `.env`:

- `DEEPGRAM_API_KEY`
- `ELEVENLABS_API_KEY`
- `SPEECHMATICS_API_KEY`

## 5) Adicionar novo modelo

1. Crie um arquivo em `src/models/` (ex.: `novo_modelo.py`).
2. Implemente uma classe com método `transcribe(audio_path: str) -> str`.
3. Registre o modelo em `src/models/__init__.py`:
   - adicione o nome em `MODELOS`.
   - adicione o mapeamento no `_map`.
4. Execute:

```bash
python3 src/benchmark.py --model NomeDoModelo
```

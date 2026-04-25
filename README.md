# stt-benchmark
 
Benchmark simples de modelos STT com WER, CER e tempo médio de inferência.
 
## Pré-requisitos
 
- **Docker** 
- **Python 3.10+** 
- Conexão à internet (para baixar dataset do Hugging Face)

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
INFO - Benchmark - Processados: 95/100 arquivos (5 pulados) — WER: 15.20%, CER: 8.45%, Tempo de inferência: 0.82s
```
 
**Nota:** O benchmark é robusto contra falhas. Se uma transcrição falhar ou a referência estiver faltando/vazia, o arquivo é pulado e o processamento continua. O resultado final mostra quantas amostras foram puladas.
 
## 3) Dataset
 
### CORAA-MUPE-ASR (padrão)
 
O benchmark usa o dataset CORAA-MUPE-ASR de fala espontânea em português brasileiro, publicado no Hugging Face:
 
- **Total de amostras** (split de teste): ~30.968
- **Após filtros** (`audio_quality=high` + duração 2–10s): ~17.854 amostras
- **Formato**: arquivos em `dataset/audio/*.wav` e `dataset/transcripts/*.txt`

**Comportamento do download:**
- **Sem `--samples`**: baixa todas as ~17.854 amostras filtradas.
- **Com `--samples N`**: baixa apenas `N` amostras. Se já houver `≥ N` arquivos locais, pula o download.

### Usar dataset próprio
 
Salve pares `.wav` + `.txt` com o mesmo nome em `dataset/audio` e `dataset/transcripts`:
 
```
dataset/
├── audio/
│   ├── amostra_1.wav
│   ├── amostra_2.wav
│   └── ...
└── transcripts/
    ├── amostra_1.txt
    ├── amostra_2.txt
    └── ...
```
 
Execute com `--samples` menor ou igual à quantidade de `.wav` locais para pular o download automático:
 
```bash
python3 src/benchmark.py --model FasterWhisper --samples 50
```
 
## 4) Modelos disponíveis
 
| Modelo | Tipo | Requer API |
|--------|------|-----------|
| `FasterWhisper` | Local | ✗ |
| `Parakeet` | Local | ✗ |
| `Deepgram` | API | ✓ |
| `Elevenlabs` | API | ✓ |
| `Speechmatics` | API | ✓ |
 
### Configurar chaves de API
 
Se usar modelos que requerem API, crie um arquivo `.env` na raiz do projeto:
 
```env
DEEPGRAM_API_KEY=sua_chave_aqui
ELEVENLABS_API_KEY=sua_chave_aqui
SPEECHMATICS_API_KEY=sua_chave_aqui
```
 
As chaves serão carregadas automaticamente via `config.py`.
 
## 5) Adicionar novo modelo
 
1. Crie um arquivo em `src/models/` (ex.: `novo_modelo.py`).
2. Implemente uma classe com método `transcribe(audio_path: str) -> str`:
```python
class NomeDoModelo:
    def __init__(self) -> None:
        logger.info("Inicializando NomeDoModelo")
    
    def transcribe(self, audio_path: str) -> str:
        # Lógica de transcrição
        return "texto transcrito"
```
 
3. Registre o modelo em `src/models/__init__.py`:
   - Adicione o nome em `MODELOS`.
   - Adicione o import e o mapeamento no `_map`.
4. Execute:
```bash
python3 src/benchmark.py --model NomeDoModelo
```

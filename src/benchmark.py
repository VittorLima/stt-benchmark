import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import time
import jiwer
import config
import models

# Logger configurado centralmente via config.py
logger = logging.getLogger("Benchmark")

# Extensões de áudio suportadas
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".opus"})

# Diretórios de entrada e saída
BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "dataset/audio"
TRANSCRIPT_DIR = BASE_DIR / "dataset/transcripts"


def main():
    # Configura o parser de argumentos para escolher o modelo a ser testado
    parser = argparse.ArgumentParser(
        description="Benchmark de modelos STT.",
    )
    parser.add_argument("--model", choices=models.MODELOS, required=True)
    args = parser.parse_args()

    # Obtém a classe do modelo selecionado dinamicamente
    model_class = getattr(models, args.model)

    # Instancia o modelo escolhido
    model = model_class()

    # Lista arquivos de áudio no diretório
    audio_files = [f for f in AUDIO_DIR.iterdir() if f.suffix in AUDIO_EXTENSIONS]
    logger.info(
        f"Encontrados {len(audio_files)} arquivos de áudio no diretório {AUDIO_DIR}"
    )

    # Variáveis para armazenar estatísticas de WER, CER e tempo de inferência
    wers, cers, inference_times = [], [], []

    # Processa cada arquivo de áudio
    for audio_file in tqdm(audio_files, desc="Processando áudio"):
        # Verifica se a transcrição de referência existe para o arquivo de áudio
        ref_path = TRANSCRIPT_DIR / f"{audio_file.stem}.txt"
        if not ref_path.exists():
            logger.warning(
                f"Transcrição de referência não encontrada para {audio_file.name}, pulando."
            )
            continue

        # Realiza a transcrição usando o modelo e mede o tempo de inferência
        start = time.perf_counter()
        hypothesis = model.transcribe(str(audio_file))
        inference_time = time.perf_counter() - start

        # Lê a transcrição de referência
        reference = ref_path.read_text(encoding="utf-8").strip()

        # Calcula WER e CER usando jiwer
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)

        # Atualiza estatísticas
        wers.append(wer)
        cers.append(cer)
        inference_times.append(inference_time)

    if not wers:
        logger.warning("Nenhum arquivo processado.")
        return

    # Calcula estatísticas agregadas e exibe resultados
    avg_wer = np.mean(wers)
    avg_cer = np.mean(cers)
    avg_inference_time = np.mean(inference_times)
    logger.info(
        f"Resultados agregados - WER: {avg_wer:.2%}, CER: {avg_cer:.2%}, Tempo de inferência: {avg_inference_time:.2f}s"
    )


if __name__ == "__main__":
    main()

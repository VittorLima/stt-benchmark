import argparse
import logging
import time
from pathlib import Path

import jiwer
import numpy as np
from tqdm import tqdm

import logging_config  # noqa: F401  (configura logging/filtro NeMo ao ser importado)
import models
from load_dataset import load_and_save_dataset
from utils import numbers_to_words

# Logger central
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "dataset/audio"
TRANSCRIPT_DIR = BASE_DIR / "dataset/transcripts"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark de modelos STT.",
    )
    parser.add_argument("--model", choices=models.MODELOS, required=True)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Limite de amostras a processar. Omitir processa tudo.",
    )
    args = parser.parse_args()

    existing_samples = len(list(AUDIO_DIR.glob("*.wav")))
    if existing_samples < (args.samples or float("inf")):
        logger.info(
            f"Encontradas {existing_samples} amostras locais. Baixando dataset..."
        )
        load_and_save_dataset(AUDIO_DIR, TRANSCRIPT_DIR, samples=args.samples)
    else:
        logger.info(f"{existing_samples} amostras já disponíveis, pulando download.")

    model_class = getattr(models, args.model)
    model = model_class()

    audio_files = list(AUDIO_DIR.glob("*.wav"))
    if args.samples is not None:
        audio_files = audio_files[: args.samples]

    wers, cers, inference_times = [], [], []
    skipped = 0

    for audio_file in tqdm(audio_files, desc="Processando áudio"):
        ref_path = TRANSCRIPT_DIR / f"{audio_file.stem}.txt"
        reference = ref_path.read_text(encoding="utf-8").strip()

        start = time.perf_counter()
        try:
            transcription = model.transcribe(str(audio_file))
            hypothesis = numbers_to_words(transcription)
        except Exception as exc:
            logger.warning(f"Transcrição falhou para {audio_file.name}, pulando: {exc}")
            skipped += 1
            continue
        inference_time = time.perf_counter() - start

        normalizer = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
        hypothesis = normalizer(hypothesis)
        reference = normalizer(reference)

        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)

        wers.append(wer)
        cers.append(cer)
        inference_times.append(inference_time)

    if not wers:
        logger.warning("Nenhum arquivo processado.")
        return

    logger.info(
        f"Processados: {len(wers)}/{len(audio_files)} arquivos "
        f"({skipped} pulados) — "
        f"WER: {np.mean(wers):.2%}, "
        f"CER: {np.mean(cers):.2%}, "
        f"Tempo de inferência: {np.mean(inference_times):.2f}s"
    )


if __name__ == "__main__":
    main()

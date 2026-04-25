import config
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import jiwer
import models
from load_dataset import load_and_save_dataset

# Logger configurado centralmente via config.py
logger = logging.getLogger("Benchmark")

# Diretórios de entrada e saída
BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "dataset/audio"
TRANSCRIPT_DIR = BASE_DIR / "dataset/transcripts"

# Cria diretórios de saída se não existirem
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Configura o parser de argumentos para escolher o modelo e limitar amostras
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

    # Baixa o dataset apenas se não houver amostras suficientes localmente
    existing_samples = len(list(AUDIO_DIR.glob("*.wav")))
    if existing_samples < (args.samples or float("inf")):
        logger.info(
            f"Encontradas {existing_samples} amostras locais. Baixando dataset..."
        )
        load_and_save_dataset(AUDIO_DIR, TRANSCRIPT_DIR, samples=args.samples)
    else:
        logger.info(f"{existing_samples} amostras já disponíveis, pulando download.")

    # Obtém a classe do modelo selecionado dinamicamente e instancia o modelo
    model_class = getattr(models, args.model)
    model = model_class()

    # Lista arquivos de áudio no diretório
    audio_files = list(AUDIO_DIR.glob("*.wav"))

    # Aplica limite de amostras também no processamento
    if args.samples is not None:
        audio_files = audio_files[: args.samples]

    # Variáveis para armazenar estatísticas de WER, CER e tempo de inferência
    wers, cers, inference_times = [], [], []
    skipped = 0  # Contador de arquivos pulados devido a erros

    # Processa cada arquivo de áudio
    for audio_file in tqdm(audio_files, desc="Processando áudio"):
        # Lê o arquivo de referência correspondente à transcrição
        ref_path = TRANSCRIPT_DIR / f"{audio_file.stem}.txt"
        reference = ref_path.read_text(encoding="utf-8").strip()

        # Realiza a transcrição usando o modelo e mede o tempo de inferência
        start = time.perf_counter()
        try:
            hypothesis = model.transcribe(str(audio_file))
        except Exception as exc:
            logger.warning(f"Transcrição falhou para {audio_file.name}, pulando: {exc}")
            skipped += 1
            continue
        inference_time = time.perf_counter() - start

        # Normalizador comum para referência e hipótese para garantir comparação justa
        normalizer = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])

        # Normaliza a hipótese e a referência antes de calcular WER e CER
        hypothesis = normalizer(hypothesis)
        reference = normalizer(reference)

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
    logger.info(
        f"Processados: {len(wers)}/{len(audio_files)} arquivos "
        f"({skipped} pulados) — "
        f"WER: {np.mean(wers):.2%}, "
        f"CER: {np.mean(cers):.2%}, "
        f"Tempo de inferência: {np.mean(inference_times):.2f}s"
    )


if __name__ == "__main__":
    main()

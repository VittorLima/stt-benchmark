import argparse
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import logging_config  # noqa: F401  (configura logging/filtro NeMo ao ser importado)
import models

logger = logging.getLogger("Latency")

BASE_DIR = Path(__file__).parent.parent
SAMPLE_AUDIO = BASE_DIR / "samples/amostra.opus"
N_RUNS = 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mede o tempo médio de inferência de um modelo STT em um áudio fixo.",
    )
    parser.add_argument("--model", choices=models.MODELOS, required=True)
    args = parser.parse_args()

    model_class = getattr(models, args.model)
    model = model_class()

    inference_times = []
    skipped = 0

    for _ in tqdm(range(N_RUNS), desc="Inferindo"):
        start = time.perf_counter()
        try:
            model.transcribe(str(SAMPLE_AUDIO))
        except Exception as exc:
            logger.warning(f"Transcrição falhou, pulando: {exc}")
            skipped += 1
            continue
        inference_times.append(time.perf_counter() - start)

    if not inference_times:
        logger.warning("Nenhuma inferência concluída.")
        return

    logger.info(
        f"Processadas: {len(inference_times)}/{N_RUNS} inferências "
        f"({skipped} puladas) — "
        f"Tempo médio de inferência: {np.mean(inference_times):.2f}s"
    )


if __name__ == "__main__":
    main()

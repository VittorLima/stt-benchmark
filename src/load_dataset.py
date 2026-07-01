import logging
from pathlib import Path

from datasets import Audio, Dataset, load_dataset


logger = logging.getLogger(__name__)


def load_and_save_dataset(
    audio_dir: Path, transcript_dir: Path, samples: int | None = None
) -> None:
    """Carrega o dataset CORAA-MUPE-ASR e salva áudio e transcrição por segmento.
        Filtros aplicados:
            - audio_quality == "high"
            - 5.0 <= duration <= 20.0 segundos

    Args:
        audio_dir: Diretório onde os arquivos de áudio serão salvos.
        transcript_dir: Diretório onde as transcrições serão salvas.
        samples: Limite de amostras a processar. None processa tudo.
    """
    try:
        logger.info("Carregando dataset CORAA-MUPE-ASR do Hugging Face...")

        ds: Dataset = load_dataset(
            "nilc-nlp/CORAA-MUPE-ASR",
            data_files={"test": "data/test-*.parquet"},
            split="test",
            verification_mode="no_checks",
        )

        # decode=False evita decodificar o áudio, preservando o formato original
        ds = ds.cast_column("audio", Audio(decode=False))

        ds_filtered = ds.filter(
            lambda sample: sample["audio_quality"] == "high"
            and 5.0 <= sample["duration"] <= 20.0
        )
        ds_filtered = ds_filtered.shuffle(seed=42)

        saved_samples = 0
        for idx, sample in enumerate(ds_filtered):
            if samples is not None and saved_samples >= samples:
                break

            try:
                audio_bytes: bytes | None = sample["audio"]["bytes"]
                transcript: str = sample["normalized_text"].strip()

                if not audio_bytes or not transcript:
                    logger.warning(
                        f"[segment_{idx}] Áudio ou transcrição vazio, pulando."
                    )
                    continue

                audio_path = audio_dir / f"segment_{idx}.wav"
                transcript_path = transcript_dir / f"segment_{idx}.txt"

                if not audio_path.exists():
                    audio_path.write_bytes(audio_bytes)
                if not transcript_path.exists():
                    transcript_path.write_text(transcript, encoding="utf-8")

                saved_samples += 1

            except Exception:
                logger.warning(f"[segment_{idx}] Pulado.")
                continue

        logger.info(f"Processamento concluído. {saved_samples} amostras salvas.")

    except Exception:
        logger.exception("Erro ao carregar e salvar o dataset.")
        raise

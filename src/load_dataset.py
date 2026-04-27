import config
import logging
from pathlib import Path
from datasets import Audio, load_dataset

# Logger configurado centralmente via config.py
logger = logging.getLogger("LoadDataset")


def load_and_save_dataset(
    audio_dir: Path, transcript_dir: Path, samples: int | None = None
) -> None:
    """Carrega o dataset CORAA-MUPE-ASR e salva áudio e transcrição por segmento.
        Filtros aplicados:
            - audio_quality == "high"
            - 5.0 <= duration <= 20.0 segundos

    Args:
        audio_dir (Path): Diretório onde os arquivos de áudio serão salvos.
        transcript_dir (Path): Diretório onde as transcrições serão salvas.
        samples (int | None): Limite de amostras a processar. None processa tudo.
    """
    try:
        logger.info("Carregando dataset CORAA-MUPE-ASR do Hugging Face...")

        # Carrega o dataset usando a biblioteca Hugging Face Datasets
        ds = load_dataset(
            "nilc-nlp/CORAA-MUPE-ASR",
            data_files={"test": "data/test-*.parquet"},
            split="test",
            verification_mode="no_checks",
        )

        # Não decodifica o áudio para manter o formato original e evitar erros de leitura
        ds = ds.cast_column("audio", Audio(decode=False))

        # Aplica os filtros de qualidade e duração
        ds_filtered = ds.filter(
            lambda sample: sample["audio_quality"] == "high"
            and 5.0 <= sample["duration"] <= 20.0
        )

        # Randomiza a ordem das amostras filtradas
        ds_filtered = ds_filtered.shuffle(seed=42)

        # Processa cada amostra filtrada, salvando o áudio e a transcrição correspondente
        saved_samples = 0  # Contador para amostras salvas
        for idx, sample in enumerate(ds_filtered):
            # Interrompe se atingiu o limite definido pelo chamador
            if samples is not None and saved_samples >= samples:
                break

            try:
                # Extrai o áudio bruto e a transcrição normalizada
                audio_bytes = sample["audio"]["bytes"]
                transcript = sample["normalized_text"].strip()

                # Pula se áudio ou transcrição estiverem vazios
                if not audio_bytes or not transcript:
                    logger.warning(
                        f"[segment_{idx}] Áudio ou transcrição vazio, pulando."
                    )
                    continue

                # Define os caminhos de saída para o áudio e a transcrição
                audio_path = audio_dir / f"segment_{idx}.wav"
                transcript_path = transcript_dir / f"segment_{idx}.txt"

                # Salva o áudio bruto como .wav (pula se já existir)
                if not audio_path.exists():
                    audio_path.write_bytes(audio_bytes)

                # Salva a transcrição normalizada com o mesmo nome (pula se já existir)
                if not transcript_path.exists():
                    transcript_path.write_text(transcript, encoding="utf-8")

                saved_samples += 1

            except Exception as e:
                # Se ocorrer um erro ao processar um segmento, loga o erro e continua com o próximo
                logger.warning(f"[segment_{idx}] Pulado: {e}")
                continue

        logger.info(f"Processamento concluído. {saved_samples} amostras salvas.")

    except Exception as exc:
        logger.error(f"Erro ao carregar e salvar o dataset: {exc}")
        raise

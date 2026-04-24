import config
import logging
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Logger configurado centralmente via config.py
logger = logging.getLogger("FasterWhisper")


class FasterWhisper:
    """Classe para transcrição de áudio usando Faster Whisper."""

    def __init__(self) -> None:
        try:
            logger.info("Inicializando FasterWhisper")

            # Carrega o modelo Faster Whisper com as configurações especificadas
            model = WhisperModel(
                model_size_or_path="large-v3-turbo",
                device="cuda",
                compute_type="float16",
            )

            # Habilita processamento em batch dos chunks internos de cada áudio
            self.model = BatchedInferencePipeline(model=model)

            logger.info("FasterWhisper carregado")

        except Exception as exc:
            logger.error(f"Erro ao inicializar FasterWhisper: {exc}")
            raise

    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio usando o modelo Faster Whisper.

        Args:
            audio_path (str): Caminho para o arquivo de áudio a ser transcrito.

        Returns:
            str: Texto transcrito do áudio.
        """
        try:
            logger.debug(f"Iniciando transcrição: {audio_path}")

            # Inferência
            segments, _ = self.model.transcribe(
                audio_path, vad_filter=False, language="pt"  # Desativa o filtro VAD
            )

            # Concatena os segmentos transcritos em uma única string
            segments = list(segments)
            transcription = " ".join([segment.text for segment in segments])

            logger.debug(f"Transcrição concluída para {audio_path}")
            return transcription

        except Exception as exc:
            logger.error(f"Erro ao transcrever áudio: {exc}")
            raise

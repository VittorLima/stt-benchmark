import config
import nemo.collections.asr as nemo_asr
import logging

# Logger configurado centralmente via config.py
logger = logging.getLogger("Parakeet")


class Parakeet:
    """Classe para transcrição de áudio usando NeMo Parakeet."""

    def __init__(self) -> None:
        try:
            logger.info("Inicializando Parakeet")

            # Carrega modelo NeMo Parakeet
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3"
            )

            logger.info("Parakeet carregado")

        except Exception as exc:
            logger.error(f"Erro ao inicializar Parakeet: {exc}")
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
            results = self.model.transcribe(audio_path)

            # Extrai transcrição do resultado
            transcription = results[0].text

            logger.debug(f"Transcrição concluída para {audio_path}")
            return transcription

        except Exception as exc:
            logger.error(f"Erro ao transcrever áudio: {exc}")
            raise

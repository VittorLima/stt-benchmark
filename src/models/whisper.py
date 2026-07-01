import logging

import whisper

logger = logging.getLogger("Whisper")


class Whisper:
    """Classe para transcrição de áudio usando o Whisper oficial da OpenAI."""

    def __init__(self) -> None:
        try:
            logger.info("Inicializando Whisper")

            self.model = whisper.load_model(
                name="turbo",
                device="cuda",
            )

            logger.info("Whisper carregado")

        except Exception as exc:
            logger.error(f"Erro ao inicializar Whisper: {exc}")
            raise

    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio usando Whisper oficial."""

        try:
            logger.debug(f"Iniciando transcrição: {audio_path}")

            result = self.model.transcribe(
                audio=audio_path,
                language="pt",
                fp16=True,
                verbose=False,
            )

            transcription = result["text"].strip()

            logger.debug(f"Transcrição concluída para {audio_path}")
            return transcription

        except Exception as exc:
            logger.error(f"Erro ao transcrever áudio: {exc}")
            raise

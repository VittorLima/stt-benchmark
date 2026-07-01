import logging

import speechmatics
from speechmatics.models import (
    ConnectionSettings,
    ServerMessageType,
    TranscriptionConfig,
)

from config import settings

logger = logging.getLogger("Speechmatics")


class Speechmatics:
    """Classe para transcrição de áudio usando Speechmatics via SDK."""

    def __init__(self) -> None:
        logger.info("Configurando Speechmatics")

        # Configurações de conexão — stateless, definidas uma única vez
        self.settings = ConnectionSettings(
            url="wss://eu.rt.speechmatics.com/v2",
            auth_token=settings.speechmatics_api_key,
        )

        self.transcription_config = TranscriptionConfig(
            language="pt",
            operating_point="enhanced",
            max_delay=5,
            enable_partials=False,
        )

        logger.info("Configuração do Speechmatics completa")

    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio usando o Speechmatics.

        Args:
            audio_path (str): Caminho para o arquivo de áudio a ser transcrito.

        Returns:
            str: Texto transcrito do áudio.
        """
        try:
            logger.debug(f"Iniciando transcrição: {audio_path}")

            self.client = speechmatics.client.WebsocketClient(self.settings)

            transcripts = []

            self.client.add_event_handler(
                event_name=ServerMessageType.AddTranscript,
                event_handler=lambda msg: transcripts.append(
                    msg["metadata"]["transcript"]
                ),
            )

            with open(audio_path, "rb") as f:
                self.client.run_synchronously(f, self.transcription_config)

            transcription = " ".join(transcripts).strip()

            logger.debug(f"Transcrição concluída para {audio_path}")
            return transcription

        except Exception as exc:
            logger.error(f"Erro ao transcrever áudio: {exc}")
            raise

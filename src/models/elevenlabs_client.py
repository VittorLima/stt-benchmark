import config
import logging
import requests

# Logger configurado centralmente via config.py
logger = logging.getLogger("Elevenlabs")


class Elevenlabs:
    """Classe para transcrição de áudio usando Elevenlabs Scribe v2."""

    def __init__(self) -> None:
        logger.info("Configurando Elevenlabs")

        self.API_URL = "https://api.elevenlabs.io/v1/speech-to-text"

        self.headers = {"xi-api-key": config.ELEVENLABS_API_KEY}

        logger.info("Configuração do Elevenlabs completa")

    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio usando o Elevenlabs Scribe v2.

        Args:
            audio_path (str): Caminho para o arquivo de áudio a ser transcrito.

        Returns:
            str: Texto transcrito do áudio.
        """
        try:
            logger.debug(f"Iniciando transcrição: {audio_path}")

            # Envia o arquivo de áudio para a API do ElevenLabs
            with open(audio_path, "rb") as f:
                response = requests.post(
                    self.API_URL,
                    headers=self.headers,
                    files={"file": f},
                    data={
                        "model_id": "scribe_v2",
                        "language_code": "pt",
                    },
                )

            # Verifica se a resposta foi bem-sucedida
            response.raise_for_status()

            # Extrai transcrição do resultado
            transcription = response.json()["text"]

            logger.debug(f"Transcrição concluída para {audio_path}")
            return transcription

        except Exception as exc:
            logger.error(f"Erro ao transcrever áudio: {exc}")
            raise

import config
import logging
import requests

# Logger configurado centralmente via config.py
logger = logging.getLogger("Deepgram")


class Deepgram:
    """Classe para transcrição de áudio usando Deepgram via HTTP."""

    def __init__(self) -> None:
        logger.info("Configurando Deepgram")

        self.API_URL = "https://api.deepgram.com/v1/listen"

        self.headers = {
            "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
            "Content-Type": "audio/*",
        }

        self.params = {
            "model": "nova-3",
            "language": "pt",
            "smart_format": "true",
        }

        logger.info("Configuração Deepgram completa")

    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio usando o Deepgram.

        Args:
            audio_path (str): Caminho para o arquivo de áudio a ser transcrito.

        Returns:
            str: Texto transcrito do áudio.
        """
        try:
            logger.debug(f"Iniciando transcrição: {audio_path}")

            # Envia o arquivo de áudio para a API do Deepgram
            with open(audio_path, "rb") as f:
                response = requests.post(
                    self.API_URL,
                    headers=self.headers,
                    params=self.params,
                    data=f,
                )

            # Verifica se a resposta foi bem-sucedida
            response.raise_for_status()

            # Extrai transcrição do resultado
            data = response.json()["results"]["channels"][0]["alternatives"][0]
            transcription = data["transcript"]

            logger.debug(f"Transcrição concluída para {audio_path}")
            return transcription

        except Exception as exc:
            logger.error(f"Erro ao transcrever áudio: {exc}")
            raise

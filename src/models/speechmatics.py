import asyncio
import json
import subprocess
import config
import logging
import websockets
import socket

# Logger configurado centralmente via config.py
logger = logging.getLogger("Speechmatics")


class Speechmatics:
    """Classe para transcrição de áudio usando Speechmatics via WebSocket."""

    def __init__(self) -> None:
        logger.info("Configurando Speechmatics")

        self.API_URL = "wss://eu.rt.speechmatics.com/v2"
        self.CHUNK_SIZE = 8192  # bytes por envio

        self.headers = {"Authorization": f"Bearer {config.SPEECHMATICS_API_KEY}"}

        self.start_recognition = {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
            },
            "transcription_config": {
                "language": "pt",
                "operating_point": "enhanced",
                "max_delay": 5,
                "enable_partials": False,
            },
        }

        logger.info("Configuração Speechmatics completa")

    def _convert_to_raw(self, audio_path: str) -> bytes:
        """Converte o áudio para PCM 16-bit 16kHz mono via ffmpeg."""
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio_path,
                "-f",
                "s16le",  # PCM 16-bit little-endian
                "-ar",
                "16000",  # 16kHz
                "-ac",
                "1",  # mono
                "-",  # saída para stdout
            ],
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Erro ao converter áudio: {result.stderr.decode()}")

        return result.stdout

    async def _transcribe_async(self, audio_path: str) -> str:
        """Lógica assíncrona de transcrição via WebSocket."""
        transcripts = []

        # Converte áudio antes de abrir o WebSocket
        audio_bytes = self._convert_to_raw(audio_path)

        async with websockets.connect(
            self.API_URL,
            additional_headers=self.headers,
            family=socket.AF_INET,  # força IPv4
        ) as ws:
            # Envia configuração de início de sessão
            await ws.send(json.dumps(self.start_recognition))

            # Aguarda confirmação do servidor para iniciar o envio de áudio
            while True:
                response = json.loads(await ws.recv())
                if response.get("message") == "RecognitionStarted":
                    break
                elif response.get("message") == "Error":
                    raise RuntimeError(f"Erro ao iniciar sessão: {response}")

            # Função para enviar áudio em chunks
            async def send_audio():
                # Envia os bytes convertidos em chunks
                for i in range(0, len(audio_bytes), self.CHUNK_SIZE):
                    chunk = audio_bytes[i : i + self.CHUNK_SIZE]
                    await ws.send(chunk)

                # Sinaliza fim do áudio
                await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))

            # Função para receber transcrições em tempo real
            async def receive_transcripts():
                async for raw in ws:
                    msg = json.loads(raw)
                    message_type = msg.get("message")

                    if message_type == "AddTranscript":
                        text = msg["metadata"]["transcript"]
                        if text.strip():
                            transcripts.append(text)

                    elif message_type == "EndOfTranscript":
                        break

                    elif message_type == "Error":
                        raise RuntimeError(f"Erro do Speechmatics: {msg}")

            # Executa envio e recebimento em paralelo
            await asyncio.gather(send_audio(), receive_transcripts())

        return " ".join(transcripts).strip()

    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio usando o Speechmatics via WebSocket.

        Args:
            audio_path (str): Caminho para o arquivo de áudio a ser transcrito.

        Returns:
            str: Texto transcrito do áudio.
        """
        try:
            logger.debug(f"Iniciando transcrição: {audio_path}")

            # Executa a lógica assíncrona de transcrição e aguarda o resultado
            transcription = asyncio.run(self._transcribe_async(audio_path))

            logger.debug(f"Transcrição concluída para {audio_path}")
            return transcription

        except Exception as exc:
            logger.error(f"Erro ao transcrever áudio: {exc}")
            raise

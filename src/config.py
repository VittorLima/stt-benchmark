import os
import sys
import re
import logging
from pathlib import Path
from typing import TextIO
from dotenv import load_dotenv


# =============================================================================
# FILTRO DE CONSOLE (REMOVE APENAS [NeMo I/W ...])
# =============================================================================
class _NeMoFilter:
    _pattern = re.compile(r"^\[NeMo\s")

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, msg: str) -> int:
        if not self._pattern.match(msg):
            return self._stream.write(msg)
        return 0

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


sys.stdout = _NeMoFilter(sys.stdout)
sys.stderr = _NeMoFilter(sys.stderr)

# =============================================================================
# carrega variáveis de ambiente do .env
# =============================================================================
load_dotenv()

# =============================================================================
# Logging centralizado
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _configure_logging() -> None:
    """
    Configura logging centralizado para stdout (ideal para containers).
    Evita logs duplicados e padroniza o formato global.
    """
    level = getattr(logging, LOG_LEVEL, logging.INFO)

    root_logger = logging.getLogger()

    # Evita duplicação de handlers
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silencia verbosidade excessiva
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)


# Inicializa logging imediatamente
_configure_logging()

# =============================================================================
# Configurações de API Keys (carregadas do .env)
# =============================================================================
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

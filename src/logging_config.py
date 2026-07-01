import json
import logging
import re
import sys
from typing import TextIO

from config import settings


class _StreamFilter:
    """Envolve um stream e descarta linhas que casem com `pattern`.

    Usado para libs (ex: NeMo) que escrevem avisos diretamente em
    stdout/stderr por fora do logging padrão, então `logger.setLevel`
    sozinho não é suficiente para silenciá-las.
    """

    def __init__(self, stream: TextIO, pattern: re.Pattern[str]) -> None:
        self._stream = stream
        self._pattern = pattern

    def write(self, msg: str) -> int:
        if self._pattern.match(msg):
            return 0
        return self._stream.write(msg)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


_NEMO_CONSOLE_PATTERN = re.compile(r"^\[NeMo\s")


def _silence_nemo_console_output() -> None:
    """Descarta as linhas `[NeMo I/W ...]` escritas direto em stdout/stderr."""
    if not isinstance(sys.stdout, _StreamFilter):
        sys.stdout = _StreamFilter(sys.stdout, _NEMO_CONSOLE_PATTERN)
    if not isinstance(sys.stderr, _StreamFilter):
        sys.stderr = _StreamFilter(sys.stderr, _NEMO_CONSOLE_PATTERN)


class JSONFormatter(logging.Formatter):
    """Formatter que produz logs em formato JSON estruturado."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Adiciona informações de exceção se houver
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Adiciona campos extras se houver
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data, ensure_ascii=False)


def _configure_logging() -> None:
    """Configura o logging para usar JSONFormatter e enviar logs para stdout.
    O nível de log é configurável via settings.log_level.
    """
    level = getattr(logging, settings.log_level, logging.INFO)
    root_logger = logging.getLogger()

    # Evita duplicação de handlers se a função for chamada múltiplas vezes
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    # Configura handler para stdout com JSONFormatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    # Configura o logger raiz
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def _silence_noisy_loggers() -> None:
    """Eleva o nível de loggers externos verbosos para evitar poluição nos logs."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)


# Inicializa logging ao importar o módulo
_silence_nemo_console_output()
_configure_logging()
_silence_noisy_loggers()

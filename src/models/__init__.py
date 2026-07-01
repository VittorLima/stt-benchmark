"""Módulo para modelos de transcrição de áudio."""

import importlib
from typing import Any

MODELOS = [
    "Deepgram",
    "Elevenlabs",
    "Parakeet",
    "FasterWhisper",
    "Speechmatics",
    "Whisper",
]


# Importação dinâmica de modelos para evitar dependências desnecessárias
def __getattr__(name: str) -> Any:
    _map = {
        "Deepgram": (".deepgram_client", "Deepgram"),
        "Speechmatics": (".speechmatics_client", "Speechmatics"),
        "Elevenlabs": (".elevenlabs_client", "Elevenlabs"),
        "Parakeet": (".parakeet", "Parakeet"),
        "FasterWhisper": (".faster-whisper", "FasterWhisper"),
        "Whisper": (".whisper", "Whisper"),
    }

    if name not in _map:
        raise AttributeError(f"Modelo não encontrado: {name}")

    module_path, class_name = _map[name]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, class_name)

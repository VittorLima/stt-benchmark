"""Módulo para modelos de transcrição de áudio."""


# Implementação de carregamento dinâmico de modelos para evitar dependências desnecessárias
def __getattr__(name):
    import importlib

    # Mapeamento de nomes de modelos para seus módulos e classes correspondentes
    _map = {
        "Deepgram": (".deepgram_client", "Deepgram"),
        "Speechmatics": (".speechmatics_client", "Speechmatics"),
        "Elevenlabs": (".elevenlabs_client", "Elevenlabs"),
        "Parakeet": (".parakeet", "Parakeet"),
        "FasterWhisper": (".whisper", "FasterWhisper"),
    }

    # Verifica se o nome do modelo solicitado está no mapeamento
    if name not in _map:
        raise AttributeError(f"Modelo não encontrado: {name}")

    # Importa o módulo correspondente e retorna a classe do modelo
    module_path, class_name = _map[name]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, class_name)

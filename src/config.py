from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    log_level: str = "INFO"
    elevenlabs_api_key: str | None = None
    speechmatics_api_key: str | None = None
    deepgram_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env")


# Inicializa e falha cedo se faltar algo
try:
    settings = Settings()
except ValidationError as e:
    missing = [err["loc"][0] for err in e.errors()]
    raise EnvironmentError(
        f"Variáveis de ambiente obrigatórias não definidas: {missing}"
    )

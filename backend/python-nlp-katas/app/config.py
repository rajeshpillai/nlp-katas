from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:3000"]
    sandbox_timeout_seconds: int = 30

    class Config:
        env_prefix = "NLP_KATAS_"


settings = Settings()

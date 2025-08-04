from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    GRADIENTS_API_KEY: str
    GRADIENTS_API_URL: str = "https://api.gradients.io"
    WANDB_ENTITY: str
    WANDB_API_KEY: str
    CHECK_INTERVAL: int = 600  # seconds
    HF_USERNAME: str
    HF_TOKEN: str

    S3_COMPATIBLE_ENDPOINT: str | None = None
    S3_COMPATIBLE_ACCESS_KEY: str | None = None
    S3_COMPATIBLE_SECRET_KEY: str | None = None
    S3_BUCKET_NAME: str | None = None

    # Model loading configuration
    USE_CPU_FOR_MODELS: bool = False

    class Config:
        env_file = ".env"


settings = Settings()

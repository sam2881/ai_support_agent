from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    # GitHub
    GITHUB_TOKEN: str
    GITHUB_REPO: str
    REPO_OWNER: str
    REPO_NAME: str

    # OpenAI
    OPENAI_API_KEY: str

    # Neo4j
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str

    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # PostgreSQL (used internally by Airflow)
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str

    # Airflow API access
    AIRFLOW_API_BASE: str  # e.g. http://localhost:8080/api/v1
    AIRFLOW_API_USERNAME: str      # e.g. admin
    AIRFLOW_API_PASSWORD: str       # e.g. admin

    # Add this new field:
    LLM_MODEL: str 

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

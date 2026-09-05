"""
config.py — Central configuration loaded from .env
All modules import from here; never read os.environ directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class DBConfig:
    # Supports a full DATABASE_URL (Supabase / any hosted Postgres)
    # or individual host/port/db/user/password vars as fallback
    database_url: str = os.getenv("DATABASE_URL", "")

    # Individual fields (used only if DATABASE_URL is not set)
    host: str     = os.getenv("POSTGRES_HOST", "localhost")
    port: int     = int(os.getenv("POSTGRES_PORT", 5432))
    db: str       = os.getenv("POSTGRES_DB", "postgres")
    user: str     = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")

    @classmethod
    def dsn(cls) -> str:
        if cls.database_url:
            return cls.database_url
        return (
            f"postgresql://{cls.user}:{cls.password}"
            f"@{cls.host}:{cls.port}/{cls.db}"
        )


class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "phi3:mini")


class EmbeddingConfig:
    model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


class RAGConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 350))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50))
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", 6))


class QAConfig:
    target_count: int = int(os.getenv("QA_TARGET_COUNT", 300))
    output_file: str = os.getenv("QA_OUTPUT_FILE", "data/qa_pairs/qa_pairs.jsonl")

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from repository root if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "").strip()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite").strip()


def get_serper_api_key() -> str:
    """Return configured Serper API key or raise descriptive configuration error."""
    key = os.getenv("SERPER_API_KEY", SERPER_API_KEY).strip()
    if not key:
        raise ValueError(
            "SERPER_API_KEY is not configured. Please set it in your environment or .env file."
        )
    return key


def get_groq_api_key() -> str:
    """Return configured Groq API key or raise descriptive configuration error."""
    key = os.getenv("GROQ_API_KEY", GROQ_API_KEY).strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Please set it in your environment or .env file."
        )
    return key


def get_groq_model() -> str:
    """Return configured Groq model identifier."""
    return os.getenv("GROQ_MODEL", GROQ_MODEL).strip() or "openai/gpt-oss-20b"

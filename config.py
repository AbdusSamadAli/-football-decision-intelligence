import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))


FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")


DATABASE_URL = os.getenv(
    "DATABASE_URL", f"sqlite:///{PROJECT_DIR / 'coaching.db'}"
)


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def require_groq_key() -> str:
    """Fail fast with a clear message if the Groq key is missing."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_here":
        raise RuntimeError(
            "Missing GROQ_API_KEY in .env. Get a free key at "
            "https://console.groq.com/keys"
        )
    return GROQ_API_KEY

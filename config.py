import os
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Project paths ===
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

RAW_JOBS = DATA_DIR / "raw_jobs.jsonl"
CLEAN_JOBS = DATA_DIR / "clean_jobs.parquet"
RESUME_TXT = DATA_DIR / "resume.txt"

JOB_EMBEDDINGS = MODELS_DIR / "job_embeddings.npy"
RESUME_EMBEDDING = MODELS_DIR / "resume_embedding.npy"

# === OpenAI Settings ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))

# === Scraping defaults ===
USER_AGENT = os.getenv(
    "USER_AGENT",
    "CS325-job-matcher/1.0 (+contact: student@school.edu)"
)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# === Ranking defaults ===
TOP_N = int(os.getenv("TOP_N", "10"))

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

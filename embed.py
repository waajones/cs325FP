# src/stage3_embed.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    BATCH_SIZE,
    CLEAN_JOBS,
    JOB_EMBEDDINGS,
    RESUME_TXT,
    RESUME_EMBEDDING,
)

def batch_embed(client: OpenAI, texts, model: str, batch_size: int):
    """Return an (N, D) float32 numpy array of embeddings for `texts`."""
    vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding jobs"):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([item.embedding for item in resp.data])
    return np.asarray(vecs, dtype=np.float32)

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env or environment."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load cleaned job data
    df = pd.read_parquet(CLEAN_JOBS)

    # Compose text to embed: compact but informative
    job_texts = (
        df["title"].fillna("") + " | " +
        df["company"].fillna("") + " | " +
        df.get("location_norm", df.get("location", "")).fillna("") + " | " +
        df.get("description_clean", df.get("description_raw", "")).fillna("")
    ).tolist()

    # Embed jobs
    X = batch_embed(client, job_texts, model=OPENAI_MODEL, batch_size=BATCH_SIZE)
    os.makedirs(JOB_EMBEDDINGS.parent, exist_ok=True)
    np.save(JOB_EMBEDDINGS, X)

    # Embed resume
    with open(RESUME_TXT, "r", encoding="utf-8") as f:
        resume_text = f.read()
    r = client.embeddings.create(model=OPENAI_MODEL, input=[resume_text]).data[0].embedding
    np.save(RESUME_EMBEDDING, np.asarray(r, dtype=np.float32))

    print(f"Saved {len(job_texts)} job vectors -> {JOB_EMBEDDINGS}")
    print(f"Saved resume vector -> {RESUME_EMBEDDING}")

if __name__ == "__main__":
    main()

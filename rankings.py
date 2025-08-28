# src/stage4_rank.py
import numpy as np
import pandas as pd
from config import (
    CLEAN_JOBS,
    JOB_EMBEDDINGS,
    RESUME_EMBEDDING,
    TOP_N as DEFAULT_TOP_N,
)

def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row in `a` and each row in `b`.
    Returns an (a_rows, b_rows) matrix.
    """
    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def rank_jobs(top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    # Load artifacts
    df = pd.read_parquet(CLEAN_JOBS)
    X = np.load(JOB_EMBEDDINGS)  # shape: (num_jobs, dim)
    r = np.load(RESUME_EMBEDDING)  # shape: (dim,) or (1, dim)

    if r.ndim == 1:
        r = r[None, :]  # (1, dim)

    # Similarity (resume vs all jobs) → (1, num_jobs)
    sims = _cosine_sim_matrix(r, X).ravel()

    # Add scores and return top N with key fields
    df = df.assign(similarity=sims)
    cols = ["title", "company", "location_norm", "url", "similarity"]
    # Fall back to 'location' if 'location_norm' missing
    cols = [c if c in df.columns else "location" for c in cols]
    out = df.sort_values("similarity", ascending=False).head(top_n)
    # Pretty rounding for readability
    out["similarity"] = out["similarity"].round(4)
    return out[cols]

def main():
    top = rank_jobs(DEFAULT_TOP_N)
    # Print a compact table for CLI usage
    print(top.to_markdown(index=False))

if __name__ == "__main__":
    main()

# src/parsing.py
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from config import RAW_JOBS, CLEAN_JOBS

# ---------- helpers ----------

def _clean_html(text: str | None) -> str:
    if not text:
        return ""
    # Strip HTML to text
    txt = BeautifulSoup(text, "html.parser").get_text(" ")
    # Collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _normalize_location(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip()
    # Light normalization examples; extend as needed
    s = s.replace("St. Louis", "Saint Louis")
    s = s.replace("MO", "Missouri")
    s = s.replace("IL", "Illinois")
    return s

def _normalize_posted_at(s: str | None) -> str:
    """
    Try to parse common 'posted at' strings into ISO date when possible.
    Falls back to original if ambiguous. Adjust/extend for your sources.
    """
    if not s:
        return ""
    s = s.strip()
    # If it's already ISO-like, leave it
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s
    # Example patterns: "2 days ago", "Yesterday", "Aug 25, 2025", "August 25, 2025"
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return s  # keep original if we can't parse

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

# ---------- main pipeline ----------

def run():
    if not RAW_JOBS.exists():
        raise FileNotFoundError(
            f"Raw jobs file not found at {RAW_JOBS}. "
            "Run Stage 1 first: `python -m src.getdata` or `python -m src.cli --acquire`."
        )

    # Load raw JSONL
    records = _read_jsonl(RAW_JOBS)

    # Normalize/clean fields
    cleaned = []
    for j in records:
        cleaned.append({
            "id": j.get("id"),
            "title": (j.get("title") or "").strip(),
            "company": (j.get("company") or "").strip(),
            "location": (j.get("location") or "").strip(),
            "location_norm": _normalize_location(j.get("location")),
            "posted_at": _normalize_posted_at(j.get("posted_at")),
            "url": j.get("url"),
            "description_raw": j.get("description_raw") or "",
            "description_clean": _clean_html(j.get("description_raw")),
        })

    df = pd.DataFrame(cleaned)

    # Basic NA handling
    df["company"] = df["company"].fillna("N/A")
    df["title"] = df["title"].fillna("")

    # De-duplicate (id preferred; fall back to title+company+location)
    if "id" in df.columns and df["id"].notna().any():
        df = df.sort_values(["id", "posted_at"]).drop_duplicates(subset=["id"], keep="last")
    else:
        df = (
            df.sort_values(["title", "company", "location_norm", "posted_at"])
              .drop_duplicates(subset=["title", "company", "location_norm"], keep="last")
        )

    # Persist cleaned parquet
    CLEAN_JOBS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_JOBS, index=False)

    print(f"Parsed {len(records)} raw rows → {len(df)} cleaned rows")
    print(f"Saved cleaned dataset → {CLEAN_JOBS}")

if __name__ == "__main__":
    run()

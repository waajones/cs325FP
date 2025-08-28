# src/getdata.py
import argparse
from pathlib import Path

from config import RAW_JOBS
from utils_io import save_jsonl  # move save_jsonl helper here
import requests, random, time
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "CS325-job-matcher/1.0 (+contact: student@school.edu)"}

def fetch_jobs(query="software engineer", location="Saint Louis, Missouri", pages=2):
    """Example job fetcher (placeholder). Replace with a real API/scraper."""
    results = []
    for p in range(pages):
        url = f"https://example.com/jobs?q={query}&l={location}&page={p+1}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for card in soup.select(".job-card"):
            results.append({
                "id": card.get("data-id"),
                "title": card.select_one(".title").get_text(strip=True),
                "company": card.select_one(".company").get_text(strip=True),
                "location": card.select_one(".location").get_text(strip=True),
                "posted_at": card.select_one(".date").get_text(strip=True),
                "url": card.select_one("a")["href"],
                "description_raw": card.select_one(".desc").get_text("\n", strip=True),
            })
        time.sleep(random.uniform(1.5, 3.0))  # polite rate limiting
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Fetch job postings and save as JSONL (Stage 1)."
    )
    parser.add_argument("--query", type=str, default="software engineer", help="Job search query")
    parser.add_argument("--location", type=str, default="Saint Louis, Missouri", help="Job location")
    parser.add_argument("--pages", type=int, default=2, help="Number of pages to fetch")
    parser.add_argument("--out", type=Path, default=RAW_JOBS, help=f"Output path (default: {RAW_JOBS})")

    args = parser.parse_args()

    print(f"Fetching jobs for '{args.query}' in '{args.location}' across {args.pages} pages...")
    jobs = fetch_jobs(query=args.query, location=args.location, pages=args.pages)
    print(f"Retrieved {len(jobs)} postings.")

    save_jsonl(jobs, path=args.out)
    print(f"Saved to {args.out.resolve()}")

if __name__ == "__main__":
    main()

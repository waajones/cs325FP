# src/cli.py
import argparse
import sys
import pandas as pd

from config import TOP_N as DEFAULT_TOP_N
from getdata import main as getdata_main
from parsing import run as parse_main
from embed import main as embed_main
from rankings import rank_jobs

def acquire():
    getdata_main([])

def parse_stage():
    parse_main()

def embed():
    embed_main()

def rankings(topn: int, save_path: str | None):
    df = rank_jobs(topn)
    print(df.to_markdown(index=False))
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nSaved ranked results → {save_path}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="AI Job Matcher — pipeline CLI")
    parser.add_argument("--acquire", action="store_true", help="Fetch job postings (Stage 1)")
    parser.add_argument("--parse", action="store_true", help="Parse & clean jobs (Stage 2)")
    parser.add_argument("--embed", action="store_true", help="Embed jobs + resume (Stage 3)")
    parser.add_argument("--rankings", action="store_true", help="Rank jobs by similarity (Stage 4)")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages in order: acquire → parse → embed → rankings")
    parser.add_argument("--topn", type=int, default=DEFAULT_TOP_N, help="Top-N results")
    parser.add_argument("--save", type=str, default=None, help="Optional CSV output path")

    args = parser.parse_args(argv)

    try:
        ran_any = False

        if args.all:
            ran_any = True
            acquire()
            parse_stage()
            embed()
            rankings(args.topn, args.save)

        else:
            if args.acquire:
                ran_any = True
                acquire()
            if args.parse:
                ran_any = True
                parse_stage()
            if args.embed:
                ran_any = True
                embed()
            if args.rankings:
                ran_any = True
                rankings(args.topn, args.save)

        if not ran_any:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

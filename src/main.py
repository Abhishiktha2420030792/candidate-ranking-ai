"""
main.py
-------
Entry point for the Redrob AI Candidate Ranking Challenge.

Usage:
    python src/main.py --candidates data/candidates.jsonl
    python src/main.py --candidates data/sample_candidates.json
    python src/main.py  # defaults to data/candidates.jsonl

Output:
    outputs/submission.csv   (candidate_id, rank, score, reasoning)
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running from project root OR from inside src/
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import load_candidates, build_candidate_profile_text
from ranker import rank_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redrob AI — Intelligent Candidate Ranking"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/candidates.jsonl"),
        help="Path to candidates JSON or JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission.csv"),
        help="Path for output CSV (default: outputs/submission.csv)",
    )
    return parser.parse_args()


def write_csv(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"[main] Submission written → {output_path}  ({len(results)} rows)")


def main() -> None:
    args = parse_args()

    # 1. Load candidates
    candidates = load_candidates(args.candidates)
    if not candidates:
        print("[main] ERROR: No candidates loaded. Check the file path.")
        sys.exit(1)

    # 2. Build profile texts
    print("[main] Building candidate profile texts …")
    profile_texts = [build_candidate_profile_text(c) for c in candidates]

    # 3. Rank
    results = rank_candidates(candidates, profile_texts)

    # 4. Write submission
    write_csv(results, args.output)

    # 5. Print top-10 preview
    print("\n=== Top 10 Candidates ===")
    header = f"{'Rank':>4}  {'CandidateID':<16}  {'Score':>7}  Title"
    print(header)
    print("-" * 70)
    for row in results[:10]:
        cid   = row["candidate_id"]
        rank  = row["rank"]
        score = row["score"]
        # Pull title for display (not in results dict, so look up by id)
        title = ""
        for c in candidates:
            if c.get("candidate_id") == cid:
                title = c.get("profile", {}).get("current_title", "")
                break
        print(f"{rank:>4}  {cid:<16}  {score:>7.4f}  {title}")


if __name__ == "__main__":
    main()
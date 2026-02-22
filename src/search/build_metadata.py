import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data/processed/papers_with_summary.jsonl"
OUTPUT_PATH = BASE_DIR / "data/processed/metadata.json"

print("Building metadata file...")

metadata = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        paper = json.loads(line)

        metadata.append({
            "paper_id": paper["paper_id"],
            "topic": paper["topic"],
            "summary": paper["summary"]
        })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved: {len(metadata)} records")
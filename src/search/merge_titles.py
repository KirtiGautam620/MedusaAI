import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = BASE_DIR / "data/processed/metadata.json"
CLUSTERS_PATH = BASE_DIR / "data/processed/papers_clusters.json"

print("Loading clusters to extract titles...")
with open(CLUSTERS_PATH, "r") as f:
    clusters = json.load(f)

# create a mapping of id to title
title_map = {}
for cluster_id, papers in clusters.items():
    for paper in papers:
        paper_id = paper.get("id")
        title = paper.get("title")
        if paper_id and title:
            title_map[paper_id] = title

print(f"Extracted {len(title_map)} titles.")

print("Updating metadata.json...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

updated_count = 0
for entry in metadata:
    pid = entry.get("paper_id")
    if pid in title_map:
        entry["title"] = title_map[pid]
        updated_count += 1

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Updated {updated_count} records with titles.")

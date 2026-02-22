import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/processed/papers_with_summary.jsonl"
OUT_PATH = BASE_DIR / "data/processed/embeddings.npy"

print("Loading papers...")
papers = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        papers.append(json.loads(line)["summary"])

print(f"{len(papers)} summaries loaded")

print("Loading embedding model (first time takes time)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
embeddings = model.encode(
    papers,
    show_progress_bar=True,
    batch_size=64,
    convert_to_numpy=True
)

print("Saving embeddings...")
np.save(OUT_PATH, embeddings)

print("Done! embeddings.npy created")
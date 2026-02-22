import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"

# relevance threshold
RELEVANCE_THRESHOLD = 0.20     
TOP_K = 5                       # max papers to be send to the frontend 

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


import textwrap
from nltk.tokenize import sent_tokenize

# search function
def search(query: str):
    """
    Returns only high-relevance research papers
    """

    # Encode query
    query_embedding = model.encode([query])

    # Compute similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Sort highest first
    sorted_indices = np.argsort(similarities)[::-1]

    results = []

    for idx in sorted_indices:

        score = float(similarities[idx])

        # filter low relevance papers 
        if score < RELEVANCE_THRESHOLD:
            continue

        paper = metadata[idx]
        paper_id = paper.get("paper_id", "Unknown")

        results.append({
            "paper_id": paper_id,
            "title": paper.get("title", f"Research Paper {paper_id}"),
            "summary": paper.get("summary", ""),
            "score": round(score, 3)
        })

        if len(results) == TOP_K:
            break

    return results


def summarize_topic(results):
    """
    Creates a brief summary of the search results using extractive summarization
    """
    if not results:
        return ""
    
    all_sentences = []
    for r in results:
        # sentence splitting
        sents = sent_tokenize(r["summary"])
        if sents:
            # first sentence of each paper as representative
            all_sentences.append(sents[0])
    
    # taking  top 3 representative sentences
    summary_bullets = [f"• {s.strip()}" for s in all_sentences[:3]]
    
    return "\n".join(summary_bullets)


#testing
if __name__ == "__main__":

    print("\n" + "="*60)
    print("      MEDUSA AI: SEMANTIC RESEARCH SEARCH ENGINE")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {RELEVANCE_THRESHOLD} | Dataset: {len(metadata)} papers")
    print("Note: Similarity scores (0 to 1) show how well papers match your intent.")
    print("="*60)

    while True:
        try:
            query = input("\n[Search] What research are you looking for? (or Ctrl+C): ").strip()
            if not query:
                continue
            
            if query.lower() in ["exit", "quit"]:
                break
                
            print("\nSearching...")
            results = search(query)

            if len(results) == 0:
                print("\nNo highly relevant papers found. Try simpler keywords.")
                continue

            print("\n" + "TOPIC INSIGHTS")
            print("─"*30)
            print(summarize_topic(results))
            print("─"*30)

            print(f"\nTop {len(results)} Matches:\n")

            for i, r in enumerate(results, 1):
                # Title with bold fallback
                title = r['title']
                print(f"{i}. \033[1;34m{title}\033[0m")
                print(f"   ID: {r['paper_id']} | Relevance: {r['score']}")
                
                # Wrapped summary
                wrapped_summary = textwrap.fill(r['summary'], width=85, initial_indent="   ", subsequent_indent="   ")
                print(f"{wrapped_summary}\n")
                if i < len(results):
                    print("   " + "┈"*40)

        except KeyboardInterrupt:
            print("\n\nExiting Medusa Search. Keep exploring!")
            break
        except Exception as e:
            print(f"\nUnexpected Error: {e}")
            continue
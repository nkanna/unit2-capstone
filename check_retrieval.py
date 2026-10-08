"""
check_retrieval.py — verify ChromaDB retrieval WITHOUT calling Gemini.

Shows the top-3 chunks and cosine distances for:
  Q1  (security)        — supported question
  Q2  (code review)     — supported question
  Q3  (customer)        — supported question
  Q4  (out-of-scope)    — unsupported question (should return poor scores)

Usage:  python check_retrieval.py
"""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = Path("chroma_store")
COLLECTION_NAME = "enterprise_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_N = 3

QUERIES = [
    ("Q1 [supported]  ", "What are the password requirements and MFA rules?"),
    ("Q2 [supported]  ", "How many approvals are required to merge a pull request?"),
    ("Q3 [supported]  ", "What is the SLA for a P1 customer complaint?"),
    ("Q4 [out-of-scope]", "What is the company's policy on office coffee machine maintenance?"),
]


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} chunks\n")
    print("=" * 72)

    for label, question in QUERIES:
        print(f"\n{label}: {question}")
        print("-" * 72)
        results = collection.query(
            query_texts=[question],
            n_results=TOP_N,
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            print(f"  Rank {rank} | source={meta['source']} chunk={meta['chunk_id']}"
                  f" | cosine_distance={dist:.4f}")
            # Print first 120 chars of the chunk for a sanity preview
            preview = doc[:120].replace("\n", " ")
            print(f"           preview: {preview!r}")

    print("\n" + "=" * 72)
    print("Retrieval check complete. No Gemini calls were made.")


if __name__ == "__main__":
    main()

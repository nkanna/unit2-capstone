"""
ingest.py — chunk enterprise documents, embed with Sentence-Transformers,
and store in a persistent ChromaDB collection.

Design decisions
----------------
Chunking
    Fixed-character window (CHUNK_SIZE=600) with 80-char overlap produces
    ~150 words per chunk.  We validate each chunk against the embedding
    model's ACTUAL max_seq_length (256 for all-MiniLM-L6-v2, confirmed
    from SentenceTransformer.max_seq_length — not the BERT tokenizer's
    default of 512, which is incorrect for this model).

    Observed max across all policy documents: 194 tokens, well inside the
    254-token content limit (256 minus 2 special tokens [CLS]/[SEP]).
    Chunks exceeding 252 tokens emit a WARNING; any chunk over 254 tokens
    raises an error to prevent silent truncation by the model.

Idempotency / stale removal
    Every chunk is keyed by "<filename>::chunk_<N>" and stored via upsert.
    Before upserting, _sync_collection() removes chunks whose source file
    was deleted or whose chunk index no longer exists (because the file
    shrank).  Re-running on unchanged files is safe and fast.

Usage
-----
    python ingest.py              # ingest data/documents/*.md
    python ingest.py --reset      # delete the whole collection first
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
DOCS_DIR = Path("data/documents")
CHROMA_PATH = Path("chroma_store")
COLLECTION_NAME = "enterprise_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
# Limits derived from the model (256 max_seq_length minus [CLS] and [SEP])
TOKEN_WARN_THRESHOLD = 252
TOKEN_HARD_LIMIT = 254

# ── Model / tokenizer (loaded once, lazily) ───────────────────────────────────
_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


def count_tokens(text: str) -> int:
    """Count tokens using the model's own tokenizer — the only correct way."""
    model = _get_st_model()
    return len(model.tokenizer(text, truncation=False)["input_ids"])


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split *text* into overlapping character windows.
    Validates each chunk's token count against the model's actual limit.
    Returns list of {id, document, metadata} dicts.
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            n_tokens = count_tokens(chunk)
            if n_tokens > TOKEN_HARD_LIMIT:
                raise ValueError(
                    f"Chunk {idx} from '{source}' has {n_tokens} tokens "
                    f"(limit {TOKEN_HARD_LIMIT}); reduce CHUNK_SIZE"
                )
            if n_tokens > TOKEN_WARN_THRESHOLD:
                print(
                    f"  WARN: chunk {idx} from '{source}' is {n_tokens} tokens "
                    f"(approaching model limit of {TOKEN_HARD_LIMIT})",
                    file=sys.stderr,
                )
            chunks.append({
                "id": f"{source}::chunk_{idx}",
                "document": chunk,
                "metadata": {
                    "source": source,
                    "chunk_id": idx,
                    "char_start": start,
                    "token_count": n_tokens,
                },
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_documents(docs_dir: Path) -> tuple[list[dict], set[str]]:
    """
    Read all .md/.txt files and return (all_chunks, current_source_filenames).
    """
    all_chunks: list[dict] = []
    current_sources: set[str] = set()

    paths = sorted(docs_dir.glob("*.md")) + sorted(docs_dir.glob("*.txt"))
    if not paths:
        sys.exit(f"No documents found in {docs_dir}")

    max_tokens_seen = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, path.name)
        current_sources.add(path.name)
        max_t = max(c["metadata"]["token_count"] for c in chunks)
        max_tokens_seen = max(max_tokens_seen, max_t)
        print(
            f"  {path.name}: {len(text)} chars → {len(chunks)} chunks "
            f"(max {max_t} tokens)"
        )
        all_chunks.extend(chunks)

    print(
        f"\n  Overall max chunk tokens: {max_tokens_seen} "
        f"(model limit: {TOKEN_HARD_LIMIT})"
    )

    # Sanity: no duplicate IDs within this batch
    ids = [c["id"] for c in all_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs — fix chunking logic"

    return all_chunks, current_sources


# ── Stale-chunk sync ──────────────────────────────────────────────────────────

def _sync_collection(
    collection: chromadb.Collection,
    all_chunks: list[dict],
    current_sources: set[str],
) -> int:
    """
    Remove chunks that are no longer valid:
      - chunks from source files that have been deleted
      - chunks whose index no longer exists because the file shrank

    Returns the count of deleted chunk IDs.
    """
    if collection.count() == 0:
        return 0

    existing = collection.get(include=["metadatas"])
    existing_by_source: dict[str, list[str]] = {}
    for cid, meta in zip(existing["ids"], existing["metadatas"]):
        existing_by_source.setdefault(meta["source"], []).append(cid)

    new_ids_by_source: dict[str, set[str]] = {}
    for c in all_chunks:
        src = c["metadata"]["source"]
        new_ids_by_source.setdefault(src, set()).add(c["id"])

    stale: list[str] = []

    # Sources deleted from disk
    for src, ids in existing_by_source.items():
        if src not in current_sources:
            stale.extend(ids)

    # Sources with fewer chunks (file shrank)
    for src, new_ids in new_ids_by_source.items():
        old_ids = set(existing_by_source.get(src, []))
        stale.extend(old_ids - new_ids)

    if stale:
        collection.delete(ids=stale)
        print(f"  Removed {len(stale)} stale chunk(s)")

    return len(stale)


# ── Collection setup ──────────────────────────────────────────────────────────

def build_collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Main ingest flow ──────────────────────────────────────────────────────────

def ingest(reset: bool = False) -> None:
    print(f"Loading documents from {DOCS_DIR} …")
    all_chunks, current_sources = load_documents(DOCS_DIR)

    print(f"\nConnecting to ChromaDB at {CHROMA_PATH} …")
    collection = build_collection(reset=reset)

    n_stale = _sync_collection(collection, all_chunks, current_sources)

    print(f"Upserting {len(all_chunks)} chunks into '{COLLECTION_NAME}' …")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    count = collection.count()
    print(
        f"\nDone. Collection '{COLLECTION_NAME}' now has {count} chunks "
        f"({n_stale} stale removed)."
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the entire collection first (use when changing embedding model)",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Override ChromaDB storage directory (default: chroma_store/)",
    )
    args = parser.parse_args()
    if args.chroma_path is not None:
        global CHROMA_PATH
        CHROMA_PATH = args.chroma_path
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()

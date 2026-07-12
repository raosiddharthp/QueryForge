"""
One-time (or re-run-on-change) ingestion: reads the seed corpus, chunks each
document with the content-type-aware router (chunker.py), embeds every chunk
with the self-hosted encoder (embeddings.py, ADR-003), and writes to
Firestore's corpus_chunks collection with the schema documented in the
design doc's §11.

Usage:
    python3 ingest.py --corpus-id acme-hr-corpus --content-type policy_docs data/seed_corpus/*.txt
"""
import argparse
import glob
import os

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

from chunker import chunk_document
from embeddings import embed


def ingest_file(db: firestore.Client, path: str, corpus_id: str, content_type: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_document(text, content_type)
    base_name = os.path.splitext(os.path.basename(path))[0]

    batch = db.batch()
    for i, chunk in enumerate(chunks):
        doc_id = f"{base_name}_chunk_{i:03d}"
        vector = embed(chunk["text"])
        ref = db.collection("corpus_chunks").document(doc_id)
        batch.set(ref, {
            "doc_id": doc_id,
            "corpus_id": corpus_id,
            "content_type": content_type,
            "chunk_strategy": chunk["chunk_strategy"],
            "chunk_version": chunk["chunk_version"],
            "text": chunk["text"],
            "embedding": Vector(vector),
            "metadata": {"source_uri": path},
        })
    batch.commit()
    return len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--content-type", required=True,
                         choices=["policy_docs", "runbooks", "faq", "email_slack", "spreadsheet"])
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    db = firestore.Client()
    total = 0
    for pattern in args.files:
        for path in glob.glob(pattern):
            n = ingest_file(db, path, args.corpus_id, args.content_type)
            print(f"  {path}: {n} chunks")
            total += n
    print(f"Ingested {total} chunks into corpus_id='{args.corpus_id}'")


if __name__ == "__main__":
    main()

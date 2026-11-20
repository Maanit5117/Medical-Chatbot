"""Build the Pinecone index from the PDFs in data/.

Run once:            python store_index.py
Resume after crash:  python store_index.py --start 740
Upload anyway:       python store_index.py --force   (can create duplicate vectors)
"""
import argparse
import os
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    INDEX_NAME,
    filter_to_minimal_docs,
    get_embeddings,
    load_pdf_files,
    text_split,
)


def main(start: int, batch_size: int, force: bool) -> None:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    chunks = text_split(filter_to_minimal_docs(load_pdf_files()))
    print(f"Number of text chunks: {len(chunks)}")

    embeddings = get_embeddings()
    dim = len(embeddings.embed_query("test"))  # read the real dimension from the model

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # Guard against uploading the same chunks twice (creates duplicate vectors).
    existing = pc.Index(INDEX_NAME).describe_index_stats().total_vector_count
    if existing and start == 0 and not force:
        print(
            f"Index '{INDEX_NAME}' already holds {existing} vectors. "
            "Use --start N to resume, or --force to upload anyway."
        )
        return

    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    for i in range(start, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for attempt in range(3):
            try:
                docsearch.add_documents(batch)
                break
            except Exception as e:
                print(f"batch at {i} failed (attempt {attempt + 1}): {type(e).__name__}: {e}")
                time.sleep(2)
        else:
            print(f"giving up at {i}; resume with --start {i}")
            return
        print(f"{min(i + batch_size, len(chunks))}/{len(chunks)}")
        time.sleep(1)  # short breather so the CPU isn't pegged continuously

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="chunk index to resume from")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="upload even if the index already has vectors")
    args = parser.parse_args()
    main(args.start, args.batch_size, args.force)
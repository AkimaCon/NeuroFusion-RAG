from __future__ import annotations
import argparse, pickle
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.config import get_settings
from app.ingestion.pdf_parser import parse_pdf_dir
from app.ingestion.embedder import SentenceTransformerEmbedder
from app.ingestion.indexer import index_chunks

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pdf-dir', default=None)
    ap.add_argument('--collection', default=None)
    args=ap.parse_args()
    s=get_settings()
    pdf_dir=Path(args.pdf_dir) if args.pdf_dir else s.ingestion.pdf_dir
    model=SentenceTransformer(s.embedding.model_name, device=s.embedding.device)
    client=chromadb.PersistentClient(path=str(s.chroma.persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
    collection=client.get_or_create_collection(name=args.collection or s.chroma.collection_name, metadata={'hnsw:space':s.chroma.distance_function})
    chunks=parse_pdf_dir(pdf_dir, max_chunk_tokens=s.ingestion.max_chunk_tokens, chunk_overlap_tokens=s.ingestion.chunk_overlap_tokens, min_chunk_chars=s.ingestion.min_chunk_chars)
    chunks=SentenceTransformerEmbedder(model).embed_chunks(chunks, batch_size=s.embedding.batch_size, normalize_embeddings=s.embedding.normalize_embeddings)
    index_chunks(chunks, collection, s.bm25.index_path, s.bm25.k1, s.bm25.b)
    print(f'Indexed {len(chunks)} chunks from {pdf_dir}')
if __name__=='__main__': main()

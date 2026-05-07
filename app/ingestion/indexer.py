from __future__ import annotations
import pickle, re
from dataclasses import dataclass
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.schema import DocumentChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+-]+")
def tokenize(text: str) -> list[str]: return [t.lower() for t in _TOKEN_RE.findall(text)]

@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[DocumentChunk]
    tokenized_corpus: list[list[str]]

def build_bm25(chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    toks = [tokenize(c.text) for c in chunks]
    return BM25Index(BM25Okapi(toks, k1=k1, b=b), chunks, toks)

def save_bm25(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f: pickle.dump(index, f)

def index_chunks(chunks: list[DocumentChunk], collection, bm25_path: Path, bm25_k1: float = 1.5, bm25_b: float = 0.75) -> None:
    if chunks:
        collection.upsert(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks], embeddings=[c.embedding for c in chunks], metadatas=[{**c.metadata, "document_id": c.document_id, "source_file": c.source_file, "page_number": c.page_number or 0, "domain": str(c.metadata.get('domain','general'))} for c in chunks])
    save_bm25(build_bm25(chunks, bm25_k1, bm25_b), bm25_path)

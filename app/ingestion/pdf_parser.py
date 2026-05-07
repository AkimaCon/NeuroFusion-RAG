from __future__ import annotations
import re, hashlib
from pathlib import Path
from pypdf import PdfReader
from app.schema import DocumentChunk

_WORD_RE = re.compile(r"\S+")

def _count_tokens(text: str) -> int:
    return len(_WORD_RE.findall(text))

def _chunks(words: list[str], size: int, overlap: int):
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        part = words[i:i+size]
        if part: yield " ".join(part)
        if i + size >= len(words): break

def parse_pdf(path: Path, max_chunk_tokens: int = 512, chunk_overlap_tokens: int = 64, min_chunk_chars: int = 80) -> list[DocumentChunk]:
    reader = PdfReader(str(path))
    doc_id = hashlib.sha1(path.name.encode()).hexdigest()[:16]
    out: list[DocumentChunk] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if len(text) < min_chunk_chars: continue
        words = text.split()
        for local_idx, chunk_text in enumerate(_chunks(words, max_chunk_tokens, chunk_overlap_tokens)):
            if len(chunk_text) < min_chunk_chars: continue
            chunk_id = f"{doc_id}-p{page_idx}-{local_idx}"
            out.append(DocumentChunk(document_id=doc_id, chunk_id=chunk_id, source_file=path.name, page_number=page_idx, text=chunk_text, token_count=_count_tokens(chunk_text), metadata={"source_path": str(path)}))
    return out

def parse_pdf_dir(pdf_dir: Path, **kwargs) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        all_chunks.extend(parse_pdf(pdf, **kwargs))
    return all_chunks

from __future__ import annotations
from app.ingestion.indexer import tokenize, BM25Index
from app.schema import Citation, EEGDomain

def bm25_search(index: BM25Index | None, query: str, top_k: int) -> list[Citation]:
    if index is None: return []
    scores = index.bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
    max_s = max([float(s) for _, s in ranked], default=1.0) or 1.0
    out=[]
    for i, s in ranked:
        c=index.chunks[i]
        out.append(Citation(chunk_id=c.chunk_id, document_id=c.document_id, source_file=c.source_file, page_number=c.page_number, section_title=c.section_title, text=c.text, dense_score=0.0, bm25_score=float(s), fusion_score=float(s)/max_s, domain=EEGDomain(str(c.metadata.get('domain','general'))) if str(c.metadata.get('domain','general')) in EEGDomain._value2member_map_ else EEGDomain.GENERAL, authors=c.metadata.get('authors',[]), year=c.metadata.get('year'), doi=c.metadata.get('doi')))
    return out

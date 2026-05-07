from __future__ import annotations
from app.schema import Citation, EEGDomain

def dense_search(collection, embedder, query: str, top_k: int, domain_filter=None, year_min=None, year_max=None) -> list[Citation]:
    where={}
    if domain_filter: where['domain']=domain_filter.value if hasattr(domain_filter,'value') else str(domain_filter)
    q_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    kwargs={"query_embeddings":[q_emb], "n_results": top_k, "include":["documents","metadatas","distances"]}
    if where: kwargs['where']=where
    res=collection.query(**kwargs)
    out=[]
    ids=res.get('ids',[[]])[0]; docs=res.get('documents',[[]])[0]; metas=res.get('metadatas',[[]])[0]; dists=res.get('distances',[[]])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        year=meta.get('year')
        if year_min and year and int(year)<year_min: continue
        if year_max and year and int(year)>year_max: continue
        score=max(0.0, 1.0-float(dist))
        dom=str(meta.get('domain','general'))
        out.append(Citation(chunk_id=cid, document_id=str(meta.get('document_id','')), source_file=str(meta.get('source_file','')), page_number=int(meta.get('page_number') or 0) or None, section_title=meta.get('section_title'), text=doc, dense_score=score, bm25_score=0.0, fusion_score=score, domain=EEGDomain(dom) if dom in EEGDomain._value2member_map_ else EEGDomain.GENERAL, authors=meta.get('authors',[]), year=year, doi=meta.get('doi')))
    return out

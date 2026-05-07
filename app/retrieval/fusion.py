from __future__ import annotations
from app.schema import Citation, RetrievalConfig, RetrievalStrategy

def _norm(vals):
    if not vals: return []
    lo, hi = min(vals), max(vals)
    return [1.0 if hi==lo and hi>0 else (v-lo)/(hi-lo) if hi!=lo else 0.0 for v in vals]

def fuse_results(dense: list[Citation], sparse: list[Citation], cfg: RetrievalConfig) -> list[Citation]:
    if cfg.strategy == RetrievalStrategy.DENSE_ONLY: return dense[:cfg.top_k_final]
    if cfg.strategy == RetrievalStrategy.BM25_ONLY: return sparse[:cfg.top_k_final]
    by_id: dict[str, Citation] = {c.chunk_id:c for c in dense+sparse}
    scores={cid:0.0 for cid in by_id}
    if cfg.strategy == RetrievalStrategy.RRF:
        for rank,c in enumerate(dense,1): scores[c.chunk_id]+=1/(cfg.rrf_k+rank)
        for rank,c in enumerate(sparse,1): scores[c.chunk_id]+=1/(cfg.rrf_k+rank)
    else:
        dnorm=dict(zip([c.chunk_id for c in dense], _norm([c.dense_score for c in dense])))
        snorm=dict(zip([c.chunk_id for c in sparse], _norm([c.bm25_score for c in sparse])))
        for cid in scores: scores[cid]=cfg.dense_weight*dnorm.get(cid,0)+(1-cfg.dense_weight)*snorm.get(cid,0)
    max_s=max(scores.values(), default=1.0) or 1.0
    merged=[]
    for cid,c in by_id.items():
        ds=max([x.dense_score for x in dense if x.chunk_id==cid] or [c.dense_score])
        bs=max([x.bm25_score for x in sparse if x.chunk_id==cid] or [c.bm25_score])
        merged.append(c.model_copy(update={"dense_score":ds,"bm25_score":bs,"fusion_score":scores[cid]/max_s}))
    return sorted(merged, key=lambda c:c.fusion_score, reverse=True)[:cfg.top_k_final]

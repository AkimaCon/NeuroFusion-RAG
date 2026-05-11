from app.retrieval.fusion import fuse_results
from app.schema import Citation, RetrievalConfig

def c(id, ds=0, bs=0):
    return Citation(
        chunk_id=id,
        document_id='d',
        source_file='s.pdf',
        text='abc',
        dense_score=ds,
        bm25_score=bs,
        fusion_score=min(max(ds, bs), 1.0),
    )

def test_rrf_fusion():
    out=fuse_results([c('a',.9), c('b',.8)], [c('b',bs=2), c('c',bs=1)], RetrievalConfig(top_k_final=3))
    assert {x.chunk_id for x in out} == {'a','b','c'}

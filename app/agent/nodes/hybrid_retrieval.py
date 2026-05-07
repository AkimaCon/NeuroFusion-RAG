from __future__ import annotations
from app.retrieval.dense import dense_search
from app.retrieval.sparse import bm25_search
from app.retrieval.fusion import fuse_results
from app.schema import AgentState, AgentStep, RetrievalConfig
from app.utils.timing import timed_node

@timed_node(AgentStep.HYBRID_RETRIEVAL.value)
async def hybrid_retrieval(state: AgentState, *, collection=None, embedder=None, bm25_index=None) -> AgentState:
    cfg: RetrievalConfig = state['retrieval_config']
    q=state.get('expanded_query') or state['query']
    dense=[] if collection is None or embedder is None else dense_search(collection, embedder, q, cfg.top_k_dense, cfg.domain_filter, cfg.year_min, cfg.year_max)
    sparse=bm25_search(bm25_index, q, cfg.top_k_bm25)
    fused=fuse_results(dense, sparse, cfg)
    steps=state.get('steps_executed',[]); steps.append(AgentStep.HYBRID_RETRIEVAL)
    state.update({'raw_chunks':dense+sparse,'reranked_chunks':fused,'steps_executed':steps})
    return state

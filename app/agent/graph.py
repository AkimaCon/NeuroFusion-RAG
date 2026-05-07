from __future__ import annotations
from langgraph.graph import StateGraph, END
from app.schema import AgentState, QueryRequest, QueryResponse
from app.agent.nodes.query_analysis import query_analysis
from app.agent.nodes.reranking import reranking
from app.agent.nodes.generation import generation
from app.agent.nodes.citation_validation import citation_validation
from app.agent.nodes.response_formatting import response_formatting
from app.agent.nodes.hybrid_retrieval import hybrid_retrieval

async def run_agent(request: QueryRequest, *, collection=None, embedder=None, bm25_index=None, settings=None) -> QueryResponse:
    async def retrieval_node(state):
        return await hybrid_retrieval(state, collection=collection, embedder=embedder, bm25_index=bm25_index)
    async def generation_node(state):
        return await generation(state, settings=settings)
    graph=StateGraph(AgentState)
    graph.add_node('query_analysis', query_analysis)
    graph.add_node('hybrid_retrieval', retrieval_node)
    graph.add_node('reranking', reranking)
    graph.add_node('generation', generation_node)
    graph.add_node('citation_validation', citation_validation)
    graph.add_node('response_formatting', response_formatting)
    graph.set_entry_point('query_analysis')
    graph.add_edge('query_analysis','hybrid_retrieval')
    graph.add_edge('hybrid_retrieval','reranking')
    graph.add_edge('reranking','generation')
    graph.add_edge('generation','citation_validation')
    graph.add_edge('citation_validation','response_formatting')
    graph.add_edge('response_formatting', END)
    app=graph.compile()
    state=await app.ainvoke({'request_id':request.request_id,'query':request.query,'conversation_history':request.conversation_history,'retrieval_config':request.retrieval_config,'steps_executed':[], 'latency_ms':{}, 'token_usage':{}})
    return state['response']

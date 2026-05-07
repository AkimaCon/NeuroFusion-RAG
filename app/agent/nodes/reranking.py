from __future__ import annotations
from app.schema import AgentState, AgentStep
from app.utils.timing import timed_node

@timed_node(AgentStep.RERANKING.value)
async def reranking(state: AgentState) -> AgentState:
    # MVP: fusion already ranks. Keep hook so a cross-encoder can be added later.
    state['reranked_chunks'] = sorted(state.get('reranked_chunks', []), key=lambda c: c.fusion_score, reverse=True)
    steps=state.get('steps_executed',[]); steps.append(AgentStep.RERANKING)
    state['steps_executed']=steps
    return state

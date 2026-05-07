from __future__ import annotations
from app.schema import AgentState, AgentStep, QueryResponse
from app.utils.timing import timed_node

@timed_node(AgentStep.RESPONSE_FORMATTING.value)
async def response_formatting(state: AgentState) -> AgentState:
    steps=state.get('steps_executed',[]); steps.append(AgentStep.RESPONSE_FORMATTING)
    state['steps_executed']=steps
    state['response']=QueryResponse(request_id=state['request_id'], answer=state.get('final_answer',''), citations=state.get('citations',[]), steps_executed=steps, retrieval_config_used=state['retrieval_config'], latency_ms=state.get('latency_ms',{}), token_usage=state.get('token_usage',{}))
    return state

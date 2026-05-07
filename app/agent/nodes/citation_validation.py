from __future__ import annotations
import re
from app.schema import AgentState, AgentStep
from app.utils.timing import timed_node

@timed_node(AgentStep.CITATION_VALIDATION.value)
async def citation_validation(state: AgentState) -> AgentState:
    answer=state.get('draft_answer','')
    chunks=state.get('reranked_chunks', [])
    used=sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer) if int(x)<=len(chunks)})
    citations=[chunks[i-1] for i in used] if used else chunks[:min(3,len(chunks))]
    steps=state.get('steps_executed',[]); steps.append(AgentStep.CITATION_VALIDATION)
    state.update({'final_answer':answer,'citations':citations,'steps_executed':steps})
    return state

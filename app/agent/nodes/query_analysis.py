from __future__ import annotations
from app.schema import AgentState, AgentStep, EEGDomain
from app.utils.timing import timed_node

ACRONYMS={"MI":"motor imagery","BCI":"brain-computer interface","ERP":"event-related potential","ERD":"event-related desynchronization","ERS":"event-related synchronization","SSVEP":"steady-state visual evoked potential","CSP":"common spatial patterns","EEG":"electroencephalography"}

@timed_node(AgentStep.QUERY_ANALYSIS.value)
async def query_analysis(state: AgentState) -> AgentState:
    q=state.get('query','')
    expanded=q
    for k,v in ACRONYMS.items():
        if k.lower() in q.lower() and v.lower() not in q.lower(): expanded += f" {v}"
    cfg=state.get('retrieval_config')
    if cfg and cfg.domain_filter is None:
        low=q.lower()
        if any(x in low for x in ['motor imagery','bci','brain-computer']):
            cfg=cfg.model_copy(update={'domain_filter': EEGDomain.BCI})
    steps=state.get('steps_executed',[]); steps.append(AgentStep.QUERY_ANALYSIS)
    state.update({'expanded_query':expanded,'retrieval_config':cfg,'steps_executed':steps})
    return state

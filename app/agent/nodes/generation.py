from __future__ import annotations
from openai import AsyncOpenAI
from app.schema import AgentState, AgentStep
from app.utils.timing import timed_node

SYSTEM="""You are an EEG research copilot. Answer only from the provided context. Use inline citation markers like [1], [2]. If context is insufficient, say so."""

def _context(chunks):
    return "\n\n".join(f"[{i}] {c.source_file}, page {c.page_number}: {c.text}" for i,c in enumerate(chunks,1))

@timed_node(AgentStep.GENERATION.value)
async def generation(state: AgentState, *, settings=None) -> AgentState:
    chunks=state.get('reranked_chunks', [])
    query=state.get('query','')
    if not chunks:
        answer="I do not have enough indexed paper context to answer yet. Ingest PDFs first with `python scripts/ingest_corpus.py`."
    elif settings is None or not settings.openai.api_key.get_secret_value():
        answer=f"Based on the retrieved paper chunks, the most relevant evidence is: " + " ".join([f"[{i}] {c.text[:350]}..." for i,c in enumerate(chunks[:3],1)])
    else:
        client=AsyncOpenAI(api_key=settings.openai.api_key.get_secret_value(), timeout=settings.openai.request_timeout)
        prompt=f"Question: {query}\n\nContext:\n{_context(chunks)}\n\nAnswer with citations."
        resp=await client.chat.completions.create(model=settings.openai.model, temperature=settings.openai.temperature, max_tokens=settings.openai.max_tokens, messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}])
        answer=resp.choices[0].message.content or ""
        usage=getattr(resp,'usage',None)
        if usage:
            state['token_usage']={'prompt':usage.prompt_tokens or 0,'completion':usage.completion_tokens or 0,'total':usage.total_tokens or 0}
    steps=state.get('steps_executed',[]); steps.append(AgentStep.GENERATION)
    state.update({'draft_answer':answer,'citations':chunks,'steps_executed':steps})
    return state

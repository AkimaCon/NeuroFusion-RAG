import requests
import uuid

URL = "http://localhost:8000/api/v1/query"

while True:
    q = input("\nAsk EEG Copilot: ")
    if q.lower() in {"exit", "quit", "q"}:
        break

    payload = {
        "request_id": str(uuid.uuid4()),
        "query": q,
        "retrieval_config": {
            "strategy": "rrf",
            "top_k_dense": 5,
            "top_k_bm25": 5,
            "top_k_final": 3,
            "rrf_k": 60,
            "dense_weight": 0.6,
            "domain_filter": "bci",
            "year_min": 1900,
            "year_max": 2100,
        },
        "conversation_history": [],
        "stream": False,
    }

    r = requests.post(URL, json=payload)
    r.raise_for_status()
    data = r.json()

    print("\nAnswer:\n")
    print(data["answer"])
from app.schema import QueryRequest, RetrievalConfig

def test_query_request_defaults():
    q=QueryRequest(query='What is motor imagery EEG?')
    assert q.retrieval_config.top_k_final == 5

def test_year_range_validation():
    try:
        RetrievalConfig(year_min=2025, year_max=2020)
    except ValueError:
        assert True
    else:
        assert False

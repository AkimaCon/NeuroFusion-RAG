from __future__ import annotations
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Request
from app.agent.graph import run_agent
from app.schema import QueryRequest, QueryResponse
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.embedder import SentenceTransformerEmbedder
from app.ingestion.indexer import index_chunks
from app.api.dependencies import get_settings_dep, get_embedder, get_chroma_collection, get_bm25_index

router=APIRouter()

@router.post('/query', response_model=QueryResponse)
async def query(req: QueryRequest, settings=Depends(get_settings_dep), embedder=Depends(get_embedder), collection=Depends(get_chroma_collection), bm25_index=Depends(get_bm25_index)):
    return await run_agent(req, collection=collection, embedder=embedder, bm25_index=bm25_index, settings=settings)

@router.post('/ingest')
async def ingest_pdf(request: Request, file: UploadFile = File(...), settings=Depends(get_settings_dep), embedder=Depends(get_embedder), collection=Depends(get_chroma_collection)):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, 'Upload a PDF file.')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(await file.read()); path=Path(tmp.name)
    chunks=parse_pdf(path, max_chunk_tokens=settings.ingestion.max_chunk_tokens, chunk_overlap_tokens=settings.ingestion.chunk_overlap_tokens, min_chunk_chars=settings.ingestion.min_chunk_chars)
    embedder_wrapper=SentenceTransformerEmbedder(embedder)
    chunks=embedder_wrapper.embed_chunks(chunks, batch_size=settings.embedding.batch_size, normalize_embeddings=settings.embedding.normalize_embeddings)
    index_chunks(chunks, collection, settings.bm25.index_path, settings.bm25.k1, settings.bm25.b)
    request.app.state.bm25_retriever = __import__('pickle').load(open(settings.bm25.index_path,'rb'))
    return {'status':'ok','chunks_indexed':len(chunks),'source_file':file.filename}

@router.get('/health')
async def health(request: Request):
    return {'status':'ok','chroma_docs':request.app.state.chroma_collection.count(), 'bm25_ready':request.app.state.bm25_retriever is not None}

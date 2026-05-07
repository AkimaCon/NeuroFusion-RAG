from __future__ import annotations
from fastapi import Request

def get_settings_dep(request: Request): return request.app.state.settings
def get_embedder(request: Request): return request.app.state.embedder
def get_chroma_collection(request: Request): return request.app.state.chroma_collection
def get_bm25_index(request: Request): return request.app.state.bm25_retriever

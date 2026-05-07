from __future__ import annotations
from app.schema import DocumentChunk

class SentenceTransformerEmbedder:
    def __init__(self, model): self.model = model
    def embed_texts(self, texts: list[str], batch_size: int = 32, normalize_embeddings: bool = True) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
        return [v.tolist() for v in vectors]
    def embed_chunks(self, chunks: list[DocumentChunk], **kwargs) -> list[DocumentChunk]:
        embeddings = self.embed_texts([c.text for c in chunks], **kwargs)
        return [c.model_copy(update={"embedding": e}) for c, e in zip(chunks, embeddings, strict=True)]

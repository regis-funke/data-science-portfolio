"""FastAPI wrapper: POST a question, get a grounded answer and its sources.

    uvicorn src.api:app --reload
    open http://127.0.0.1:8000/docs

The vector store and the LLM client are created once at startup, not per
request. Opening the store re-reads it from disk and constructing the client
re-reads configuration; doing either inside the handler would add that cost to
every call for no benefit.

Sources are built from the retrieved documents' metadata, never parsed out of
the model's answer text. The eval runs showed the model dropping page numbers
and, once, inventing the range "p.2-3" by merging two chunks - fine for a human
reader, useless as an API contract. Retrieval already knows exactly which
chunks it returned, so the API reports those.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from query import (  # noqa: E402
    DEFAULT_K,
    LLM_MODEL,
    PROMPT,
    format_docs,
    load_vector_store,
)

# Populated at startup by the lifespan handler below.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the store and build the chain once, before the first request.

    A missing store raises here rather than on the first call, so the failure
    shows up at startup with a clear message instead of as a 500 later.
    """
    db = load_vector_store()
    state["db"] = db
    state["chain"] = (
        PROMPT | ChatOpenAI(model=LLM_MODEL, temperature=0) | StrOutputParser()
    )
    state["vectors"] = db._collection.count()
    print(f"Loaded {state['vectors']} vectors")
    yield
    state.clear()


app = FastAPI(
    title="RAG over ML papers",
    description="Ask a question; get an answer grounded in the indexed papers, "
    "or an explicit refusal when they do not contain it.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, examples=["What are BERT's two pre-training tasks?"]
    )
    # Bounded rather than free: k is what the caller could use to run up a bill,
    # since every extra chunk is context sent to the model.
    k: int = Field(DEFAULT_K, ge=1, le=10, description="Chunks to retrieve.")


class Source(BaseModel):
    document: str
    page: str
    # Distance from the query vector, squared L2 as Chroma returns it: lower is
    # closer. Exposed because a caller judging whether to trust an answer wants
    # to see how well the retrieval actually matched.
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_ms: int
    generation_ms: int


@app.get("/health")
def health():
    """Liveness plus the one fact worth checking: the store is loaded."""
    return {"status": "ok", "vectors": state.get("vectors", 0), "model": LLM_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    db, chain = state["db"], state["chain"]

    start = time.perf_counter()
    hits = db.similarity_search_with_score(request.question, k=request.k)
    retrieval_ms = int((time.perf_counter() - start) * 1000)

    if not hits:
        raise HTTPException(status_code=503, detail="Vector store returned no results.")

    docs = [doc for doc, _ in hits]

    start = time.perf_counter()
    answer = chain.invoke({"context": format_docs(docs), "question": request.question})
    generation_ms = int((time.perf_counter() - start) * 1000)

    return AskResponse(
        answer=answer,
        sources=[
            Source(
                document=doc.metadata["source"],
                page=str(doc.metadata["page_label"]),
                score=round(float(score), 4),
            )
            for doc, score in hits
        ],
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )

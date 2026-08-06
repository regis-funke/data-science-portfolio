import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

REPO_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
CHROMA_DIR = REPO_ROOT / "chroma_db"


LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

PROMPT = ChatPromptTemplate.from_template(
    """Answer using ONLY the context below. If the context fully answers
the question, answer it. If it only partially answers, state what the
context supports and say explicitly what is missing. If it does not
address the question at all, say you don't know. Always cite the
source document.

Context:
{context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['source']} p.{d.metadata['page_label']}]\n{d.page_content}"
        for d in docs
    )


def build_chain(db, k=4):
    retriever = db.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | ChatOpenAI(model=LLM_MODEL, temperature=0)
        | StrOutputParser()
    )


@contextmanager
def timed(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        m, s = divmod(elapsed, 60)
        pretty = f"{int(m)}m {s:.0f}s" if m else f"{s:.1f}s"
        print(f"{label}: {pretty}")


def load_vector_store(persist_dir=CHROMA_DIR):
    """Reopen an existing store — no re-embedding, no cost."""
    if not persist_dir.exists():
        raise FileNotFoundError(f"No store at {persist_dir}. Run ingest.py first.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def preview(query, db, k=4, mmr=False, fetch_k=30, lambda_mult=0.8, chars=300):
    if mmr:
        with timed(f"mmr ({k=}, {fetch_k=}, {lambda_mult=})"):
            docs = db.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        results = [(d, None) for d in docs]
    else:
        with timed(f"search ({k=})"):
            results = db.similarity_search_with_score(query, k=k)

    print(f"\nQ: {query}")
    for doc, score in results:
        meta = doc.metadata
        label = f"[{score:.3f}] " if score is not None else ""
        print("-" * 60)
        print(f"{label}{meta['source']} p.{meta['page_label']}")
        print(doc.page_content[:chars].replace("\n", " ") if chars else doc.page_content)


if __name__ == "__main__":
    with timed("open store"):
        db = load_vector_store()
    print(f"{db._collection.count()} vectors")

    chain = build_chain(db)

    questions = [
        # (a) answerable from a single chunk
        "What is the maximum path length in a self-attention layer?",
        # (b) needs several chunks, answer partly hidden behind a cross-reference
        "How does BERT's pre-training objective differ from GPT's?",
        # (c) not in the corpus at all — must refuse
        "What is the capital of Portugal?",
    ]

    for q in questions:
        print("=" * 70)
        print(f"Q: {q}")
        with timed("ask"):
            print(chain.invoke(q))
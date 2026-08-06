import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

REPO_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
CHROMA_DIR = REPO_ROOT / "chroma_db"

EMBEDDING_MODEL = "text-embedding-3-small"


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


def preview(query, db, k=4):
    with timed(f"search ({k=})"):
        results = db.similarity_search_with_score(query, k=k)

    print(f"\nQ: {query}")
    for doc, score in results:
        meta = doc.metadata
        print("-" * 60)
        print(f"[{score:.3f}] {meta['source']} p.{meta['page_label']}")
        print(doc.page_content[:300].replace("\n", " "))


if __name__ == "__main__":
    with timed("open store"):
        db = load_vector_store()
    print(f"{db._collection.count()} vectors")

    preview("what problem does attention solve?", db)
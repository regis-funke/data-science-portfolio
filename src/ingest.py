import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import time
from contextlib import contextmanager


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

load_dotenv()

REPO_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = REPO_ROOT / "data"
CHROMA_DIR = REPO_ROOT / "chroma_db"

def load_documents():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {DATA_DIR.resolve()}")

    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

def clean_documents(docs):
    """Return new Documents: hyphenated line breaks rejoined, source as filename."""
    cleaned = []
    for doc in docs:
        text = unicodedata.normalize("NFKC", doc.page_content)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        metadata = {**doc.metadata, "source": Path(doc.metadata["source"]).name}
        cleaned.append(Document(page_content=text, metadata=metadata))
    return cleaned

def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    """Split page-level Documents into overlapping chunks. Metadata is carried over."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def build_vector_store(chunks, persist_dir=CHROMA_DIR):
    """Embed chunks and write them to a local Chroma DB. Costs money."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

if __name__ == "__main__":
    with timed("load"):
        docs = load_documents()
    print(f"Loaded {len(docs)} pages from {DATA_DIR.resolve()}")

    cleaned = clean_documents(docs)
    print(cleaned[0].metadata)
    print(cleaned[0].page_content[:500])

    chunks = chunk_documents(cleaned)
    print(f"{len(cleaned)} pages -> {len(chunks)} chunks")

    for chunk in chunks[:3]:
        print("-" * 60)
        print(chunk.metadata)
        print(chunk.page_content)

    if CHROMA_DIR.exists():
        raise SystemExit(f"{CHROMA_DIR} already exists — delete it to rebuild.")

    with timed("embed + store"):
        db = build_vector_store(chunks)
    print(f"Stored {db._collection.count()} vectors in {CHROMA_DIR}")
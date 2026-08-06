"""Ingestion: load PDFs -> clean -> chunk -> embed -> persist to Chroma.

Everything here happens offline and once. The expensive, paid step is
embedding; after it has run, `query.py` reads the store from disk for free.
Re-run this script only when the corpus or the chunking strategy changes.
"""

import re
import shutil
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reads OPENAI_API_KEY from .env into the environment. The OpenAI clients below
# pick it up implicitly, so the key never appears in this file.
load_dotenv()

# Resolve paths relative to the repo root so the script behaves the same whether
# it is run from the repo root, from src/, or from a notebook cell.
REPO_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = REPO_ROOT / "data"
CHROMA_DIR = REPO_ROOT / "chroma_db"

# Must match EMBEDDING_MODEL in query.py: a store built with one model can only
# be searched with the same model, since the vectors live in its space.
#
# text-embedding-3-large after the eval sweep: it scored 10/10 where the -small
# store scored 7/10 at the same chunk size and k. It was the highest-leverage
# variable by a distance - larger than chunk size or k, which neither fixed the
# two failures it fixed. It also halved the pipeline's sensitivity to how a
# question is worded: a vague query that -small answered by ranking a title page
# first is ranked correctly by -large, unrewritten.
#
# Costs ~6.5x more to embed (cents, once) and returns 3072-dimensional vectors,
# so retrieval is marginally slower and the store is larger on disk.
EMBEDDING_MODEL = "text-embedding-3-large"

# Chunk size trades retrieval precision against context completeness. Smaller
# chunks isolate a fact more sharply; larger ones keep an explanation intact.
# The overlap exists so a fact spanning a boundary survives in at least one chunk.
#
# Measured 600 against 1000 on this corpus at matched context budgets (k=7 vs
# k=4) and 1000 won both times. Dense figure and table text is the reason: it
# has no paragraph boundaries for the splitter to respect, so at 600 it gets cut
# into fragments too sparse to retrieve. BERT's Figure 1 caption - the only place
# the corpus names "NSP" and "Mask LM" - survives intact at 1000 and does not
# at 600. Whether 1500 is better still is untested.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@contextmanager
def timed(label):
    """Print how long the wrapped block took.

    perf_counter rather than time.time: it is monotonic, so a system clock
    adjustment mid-run cannot produce a negative duration. Duplicated in
    query.py; move both to src/utils.py if a third caller appears.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        minutes, seconds = divmod(elapsed, 60)
        pretty = f"{int(minutes)}m {seconds:.0f}s" if minutes else f"{seconds:.1f}s"
        print(f"{label}: {pretty}")


def load_documents():
    """Load every PDF in data/ as one Document per page.

    Sorted so the ingestion order is deterministic across runs.
    """
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {DATA_DIR.resolve()}")

    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())  # one Document per page, not per file
    return docs


def clean_documents(docs):
    """Return new Documents with PDF text artefacts repaired.

    Pure function: the input docs are untouched, so chunking can be re-run with
    different settings without re-reading the PDFs.

    Two fixes, both specific to LaTeX-produced PDFs:

    1. NFKC normalization expands typographic ligatures. LaTeX emits "fi" as the
       single character U+FB01, so "fine-tuning" is stored as "ﬁne-tuning" and a
       query for "fine-tuning" would not match it.
    2. A word split across a line break ("representa-\\ntion") is rejoined. Left
       alone it embeds as two meaningless fragments.

    Known limitation: rule 2 cannot distinguish a soft hyphen from a real one, so
    genuinely hyphenated compounds broken across lines are merged too
    ("task-specific" -> "taskspecific"). Accepted: the merged form still embeds
    near the correct one, whereas a severed word embeds near nothing.
    """
    cleaned = []
    for doc in docs:
        text = unicodedata.normalize("NFKC", doc.page_content)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Copy the dict rather than mutating it: metadata is shared by reference,
        # so editing in place would also alter the original Document.
        # Store the bare filename, not the absolute path - it is what gets shown
        # to the model as a citation, and it keeps local paths out of the output.
        metadata = {**doc.metadata, "source": Path(doc.metadata["source"]).name}

        cleaned.append(Document(page_content=text, metadata=metadata))
    return cleaned


def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split page-level Documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries separators in order ("\\n\\n", then
    "\\n", then " ") and only falls back to a hard character cut if none fit.
    That is why the newlines left by clean_documents are worth keeping: they are
    the splitter's best signal for paragraph boundaries.

    split_documents copies each page's metadata onto every chunk derived from it,
    so `source` and `page_label` survive into the citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Records each chunk's character offset within its page, which makes a
        # surprising retrieval result traceable back to the original text.
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks, persist_dir=CHROMA_DIR):
    """Embed every chunk and write the vectors to a local Chroma DB.

    Any existing store at `persist_dir` is deleted first. Chroma's
    from_documents appends rather than replaces, so without this a second run
    would leave both generations of chunks in the collection - and the stale
    ones would keep competing for the top-k slots. Deleting also guarantees the
    store on disk always matches the CHUNK_SIZE above, which is otherwise easy
    to let drift.

    This is the only step that costs money (one embedding call per chunk) and
    the only one that takes real time. Chroma persists to disk automatically.

    Note: Chroma's default distance is squared L2, not cosine. For the unit-length
    vectors OpenAI returns, cosine similarity = 1 - distance / 2, so a reported
    distance of 0.58 means a similarity of about 0.71. Lower is better.
    """
    if persist_dir.exists():
        print(f"Removing existing store at {persist_dir}")
        shutil.rmtree(persist_dir)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
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

    # Read these by eye. Chunk quality is judged by reading, not by counting.
    for chunk in chunks[:3]:
        print("-" * 60)
        print(chunk.metadata)
        print(chunk.page_content)

    # Rebuilds from scratch every run: build_vector_store deletes any existing
    # store first, so the DB on disk always reflects the settings above.
    with timed("embed + store"):
        db = build_vector_store(chunks)
    print(f"Stored {db._collection.count()} vectors in {CHROMA_DIR}")

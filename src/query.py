"""Query side: reopen the vector store, inspect retrieval, answer questions.

Two entry points:

- `preview()` runs retrieval alone, with no LLM involved. Use it to judge
  whether the right chunks come back before blaming the model for a bad answer.
- `build_chain()` composes the full retrieve-then-answer pipeline.

Requires a store built by ingest.py.
"""

import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

REPO_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
CHROMA_DIR = REPO_ROOT / "chroma_db"

# Must match EMBEDDING_MODEL in ingest.py - the question is embedded at query
# time and compared against vectors built at ingestion time, so both have to
# live in the same vector space. See the note there for why -large.
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

# How many chunks to retrieve. Measured twice on this corpus: k=6 added only
# near-duplicate chunks, and k=7 scored no better than k=4 across the full eval
# set while sending 76% more context. More context is not better retrieval - at
# k=7 the model still answered two questions wrong, because the wrong chunk was
# still ranked first.
DEFAULT_K = 4

# The groundedness prompt - the main defence against hallucination.
#
# It deliberately offers three outcomes rather than two. An earlier version had
# only "answer" or "say you don't know", which threw away partial information:
# asked to compare BERT and GPT, the model held a chunk stating GPT's objective
# and refused outright because BERT's was missing. Naming the gap is more useful
# than silence, and testing confirmed the looser wording still refuses questions
# the corpus does not address at all.
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


@contextmanager
def timed(label):
    """Print how long the wrapped block took. See the note in ingest.py."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        minutes, seconds = divmod(elapsed, 60)
        pretty = f"{int(minutes)}m {seconds:.0f}s" if minutes else f"{seconds:.1f}s"
        print(f"{label}: {pretty}")


def load_vector_store(persist_dir=CHROMA_DIR):
    """Reopen the persisted Chroma store.

    Reads existing vectors from disk - nothing is re-embedded, so this is free
    and near-instant. Only the question itself is embedded, at search time.
    """
    if not persist_dir.exists():
        raise FileNotFoundError(f"No store at {persist_dir}. Run ingest.py first.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def format_docs(docs):
    """Flatten retrieved chunks into one string for the prompt.

    Each chunk is prefixed with its source and page. Citation quality is decided
    here, not in the prompt: the model can only cite what it can see.
    """
    return "\n\n".join(
        f"[{d.metadata['source']} p.{d.metadata['page_label']}]\n{d.page_content}"
        for d in docs
    )


def build_chain(db, k=DEFAULT_K):
    """Compose the retrieve-then-answer chain in LCEL (LangChain's pipe syntax).

    The question string is fed to two things at once: the retriever, which turns
    it into formatted context, and RunnablePassthrough, which forwards it
    unchanged. The dict that results fills the prompt's two placeholders.

    Plain similarity search rather than MMR. MMR was measured on this corpus and
    hurt: it rewards dissimilarity from chunks already chosen, so it evicted the
    answering chunk in favour of unrelated material from other papers.

    temperature=0 because the same context should yield the same answer twice.
    """
    retriever = db.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | ChatOpenAI(model=LLM_MODEL, temperature=0)
        | StrOutputParser()
    )


def preview(query, db, k=DEFAULT_K, mmr=False, fetch_k=30, lambda_mult=0.8, chars=300):
    """Print the chunks a query retrieves, without calling the LLM.

    The diagnostic to reach for first: if the answer is wrong, this shows whether
    retrieval or generation is at fault.

    Watch the spread between the best and worst score more than the absolute
    values. A flat spread means the index cannot tell rank 1 from rank 4 and the
    ordering is close to noise - usually a sign of a vague query.

    Args:
        chars: characters of each chunk to print; 0 prints the chunk in full,
            which is what the model actually receives.
        mmr: use Maximal Marginal Relevance instead of plain similarity. It
            fetches `fetch_k` candidates and picks `k`, penalising each pick for
            resembling earlier ones. lambda_mult=1.0 is pure relevance
            (equivalent to plain search), 0.0 is pure diversity. Kept as an
            option for comparison; plain search is the better default here.
    """
    if mmr:
        with timed(f"mmr ({k=}, {fetch_k=}, {lambda_mult=})"):
            mmr_docs = db.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        # MMR returns documents without scores, hence the None placeholder.
        results = [(doc, None) for doc in mmr_docs]
    else:
        with timed(f"search ({k=})"):
            results = db.similarity_search_with_score(query, k=k)

    print(f"\nQ: {query}")
    for doc, score in results:
        meta = doc.metadata
        label = f"[{score:.3f}] " if score is not None else ""
        print("-" * 60)
        print(f"{label}{meta['source']} p.{meta['page_label']}")
        text = doc.page_content
        print(text[:chars].replace("\n", " ") if chars else text)


if __name__ == "__main__":
    with timed("open store"):
        db = load_vector_store()
    print(f"{db._collection.count()} vectors")

    chain = build_chain(db)

    # Three question types, each testing a different property.
    questions = [
        # (a) answerable from a single chunk - tests basic retrieval and citation
        "What is the maximum path length in a self-attention layer?",
        # (b) needs several chunks, and the answer is partly hidden behind an
        #     internal cross-reference ("see Section 3.1") that retrieval cannot
        #     follow - tests honest handling of incomplete context
        "How does BERT's pre-training objective differ from GPT's?",
        # (c) not in the corpus at all - must refuse rather than answer from the
        #     model's own knowledge
        "What is the capital of Portugal?",
    ]

    for question in questions:
        print("=" * 70)
        print(f"Q: {question}")
        with timed("ask"):
            print(chain.invoke(question))

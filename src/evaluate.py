"""Evaluation harness: run the eval set across several configurations.

Builds (and caches) one vector store per distinct chunking/embedding setting,
answers every question in eval_questions.md under each configuration, and writes
a markdown report grouping answers by question so they can be read side by side.

Usage:
    python src/evaluate.py

Named evaluate.py rather than eval.py to avoid confusion with the builtin.

Cost note: each new chunking or embedding setting re-embeds the whole corpus
(cents, not dollars). Stores are cached in eval_stores/, so re-running only pays
for configurations it has not seen. Every question costs one LLM call per
configuration.
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

sys.path.insert(0, str(Path(__file__).parent))

from ingest import chunk_documents, clean_documents, load_documents, timed  # noqa: E402
from query import PROMPT, format_docs  # noqa: E402

REPO_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
QUESTIONS_FILE = REPO_ROOT / "eval_questions.md"
STORE_CACHE = REPO_ROOT / "eval_stores"
RESULTS_DIR = REPO_ROOT / "eval_results"

# Each entry is one configuration to compare. `name` is used as the column
# heading in the report and as the cache key, so keep it short and unique.
#
# Only vary one axis at a time. Comparing 600/k=4 against 1000/k=7 confounds
# granularity with total context volume (k x chunk_size), and the result cannot
# be attributed to either.
#
# These four are the sweep of 2026-08-06, kept so the result is reproducible.
# emb-large-1000-k4 won 10/10 and is now the default in ingest.py and query.py;
# see the Results section of eval_questions.md.
CONFIGS = [
    {
        "name": "base-1000-k4",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "embedding": "text-embedding-3-small",
        "k": 4,
        "llm": "gpt-4o-mini",
    },
    {
        "name": "large-1500-k4",
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "embedding": "text-embedding-3-small",
        "k": 4,
        "llm": "gpt-4o-mini",
    },
    {
        "name": "base-1000-k7",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "embedding": "text-embedding-3-small",
        "k": 7,
        "llm": "gpt-4o-mini",
    },
    {
        "name": "emb-large-1000-k4",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "embedding": "text-embedding-3-large",
        "k": 4,
        "llm": "gpt-4o-mini",
    },
]


def load_questions(path=QUESTIONS_FILE):
    """Parse questions out of eval_questions.md.

    The markdown file is the single source of truth. Keeping a second copy of
    the questions in this script would let the two drift apart, which is exactly
    the class of bug that has already bitten this project twice (chunk size
    versus the store on disk).

    Expects headings of the form "## Q3 — ..." followed by a "**Question:**"
    line. Raises rather than silently evaluating a subset.
    """
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^## (Q\d+)\s*—\s*(.+?)$.*?^\*\*Question:\*\*\s*(.+?)(?=\n\n)",
        re.MULTILINE | re.DOTALL,
    )
    questions = [
        {
            "id": qid,
            "category": category.strip(),
            "text": " ".join(body.split()),  # collapse wrapped lines
        }
        for qid, category, body in pattern.findall(text)
    ]
    if not questions:
        raise ValueError(f"No questions parsed from {path} — check the format.")
    return questions


def get_store(chunk_size, chunk_overlap, embedding, pages):
    """Return a vector store for these settings, building it only if needed.

    Stores are cached by setting, so a sweep that varies only k reuses one store
    and costs nothing extra to embed. `pages` is the cleaned corpus, loaded once
    by the caller and shared across every configuration.
    """
    slug = f"{embedding}_{chunk_size}_{chunk_overlap}".replace("/", "-")
    persist_dir = STORE_CACHE / slug
    embeddings = OpenAIEmbeddings(model=embedding)

    if persist_dir.exists():
        print(f"  store: reusing {slug}")
        return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

    chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  store: building {slug} ({len(chunks)} chunks)")
    with timed("  embed"):
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )


def answer(question, db, k, llm_model):
    """Answer one question, returning the answer and the chunks behind it.

    Retrieval is done explicitly rather than inside an LCEL chain so the report
    can show which chunks produced each answer. A wrong answer is only
    diagnosable if the retrieved sources are recorded alongside it.

    Retrieval and generation are timed separately. They scale with different
    things - retrieval with the embedding model and index size, generation with
    how much context k puts in front of the LLM - and a single combined number
    would hide which one a configuration change actually cost.
    """
    start = time.perf_counter()
    docs = db.similarity_search(question, k=k)
    retrieval_s = time.perf_counter() - start

    chain = PROMPT | ChatOpenAI(model=llm_model, temperature=0) | StrOutputParser()
    start = time.perf_counter()
    text = chain.invoke({"context": format_docs(docs), "question": question})
    generation_s = time.perf_counter() - start

    sources = [f"{d.metadata['source']} p.{d.metadata['page_label']}" for d in docs]
    return {
        "answer": text,
        "sources": sources,
        "retrieval_s": retrieval_s,
        "generation_s": generation_s,
        # Characters of context sent to the model - the thing k actually buys,
        # and the thing it actually costs.
        "context_chars": sum(len(d.page_content) for d in docs),
    }


def run(configs=CONFIGS, questions=None):
    """Answer every question under every configuration."""
    questions = questions or load_questions()
    print(f"{len(questions)} questions x {len(configs)} configs")

    # Load and clean once: identical for every configuration, and re-reading the
    # PDFs per config would dominate the runtime for no reason.
    with timed("load + clean corpus"):
        pages = clean_documents(load_documents())

    results = {}
    for config in configs:
        print(f"\n[{config['name']}]")
        db = get_store(
            config["chunk_size"], config["chunk_overlap"], config["embedding"], pages
        )
        with timed(f"  {config['name']} answers"):
            results[config["name"]] = [
                answer(q["text"], db, config["k"], config["llm"]) for q in questions
            ]

    return questions, results


def write_report(questions, results, configs=CONFIGS, results_dir=RESULTS_DIR):
    """Write a markdown report grouped by question.

    Grouped by question, not by configuration: the point is to compare answers
    to the same question, and scrolling between sections to do that defeats the
    purpose. Grades are left blank - they are a judgement call, made by reading.
    """
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    path = results_dir / f"eval_{stamp}.md"

    lines = [f"# Eval run {stamp}", ""]

    lines += ["## Configurations", "", "| name | chunk | overlap | embedding | k |"]
    lines += ["|---|---|---|---|---|"]
    for c in configs:
        lines.append(
            f"| {c['name']} | {c['chunk_size']} | {c['chunk_overlap']} "
            f"| {c['embedding']} | {c['k']} |"
        )
    lines.append("")

    # Latency summary. Averages over the whole question set, so a single slow
    # answer does not decide the comparison.
    lines += [
        "## Cost per question (mean)",
        "",
        "| config | retrieval | generation | context chars |",
        "|---|---|---|---|",
    ]
    for c in configs:
        runs = results[c["name"]]
        n = len(runs)
        lines.append(
            f"| {c['name']} "
            f"| {sum(r['retrieval_s'] for r in runs) / n:.2f}s "
            f"| {sum(r['generation_s'] for r in runs) / n:.2f}s "
            f"| {sum(r['context_chars'] for r in runs) // n:,} |"
        )
    lines.append("")

    lines += ["## Grades", "", "| Q | " + " | ".join(c["name"] for c in configs) + " |"]
    lines.append("|---" * (len(configs) + 1) + "|")
    for q in questions:
        lines.append(f"| {q['id']} | " + " | ".join([""] * len(configs)) + " |")
    lines += ["", "_Fill in by reading the answers below: pass / partial / fail._", ""]

    for i, q in enumerate(questions):
        lines += [f"## {q['id']} — {q['category']}", "", f"**{q['text']}**", ""]
        for c in configs:
            r = results[c["name"]][i]
            timing = (
                f"retrieval {r['retrieval_s']:.2f}s · "
                f"generation {r['generation_s']:.2f}s · "
                f"{r['context_chars']:,} chars"
            )
            lines += [f"### {c['name']}", "", f"_{timing}_", "", r["answer"], ""]
            lines += ["<details><summary>retrieved</summary>", ""]
            lines += [f"- {s}" for s in r["sources"]]
            lines += ["", "</details>", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    with timed("total"):
        questions, results = run()
        report = write_report(questions, results)
    print(f"\nReport: {report}")

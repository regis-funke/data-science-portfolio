"""Phase 6: an agent that decides whether to retrieve, vs. a chain that always does.

The Phase 3 chain has retrieval wired into it structurally - every question
triggers a vector search, because that is what the pipe does. An agent is given
retrieval as a *tool* and chooses whether to call it, how to word the query, and
whether one call was enough.

Run it to see the difference:

    python src/agent.py

The comparison reports how many times each question actually hit the vector
store, which is the measurement that makes the distinction concrete rather than
theoretical.
"""

import sys
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent))

from query import DEFAULT_K, build_chain, load_vector_store  # noqa: E402

# Deliberately not the chain's gpt-4o-mini, and the difference was measured.
#
# Asked "What is the capital of Portugal?", gpt-4o-mini skipped the search and
# answered "Lisbon" - through three increasingly explicit versions of the system
# prompt below, including a flat instruction never to answer a factual question
# from its own knowledge. gpt-4o, same prompt and tools, declined.
#
# The chain does not need this. It cannot answer from memory because retrieved
# context is the only text it ever sees; its groundedness is structural. The
# agent's groundedness is an instruction, and instructions can be overridden by
# a model confident enough in what it already knows. Discretion has to be paid
# for, either in model capability or in a structural guard that forces a search.
AGENT_MODEL = "gpt-4o"

# The tool description is the agent's only basis for deciding whether to call
# it. It is a prompt, not documentation - vague wording here shows up as the
# agent retrieving when it shouldn't, or answering from its own knowledge when
# it should have searched. Naming the corpus explicitly is what lets it tell
# "what does BERT mask?" from "how are you?".
TOOL_DESCRIPTION = (
    "Search the text of six machine learning papers: Attention Is All You Need, "
    "BERT, GPT-3 (Language Models are Few-Shot Learners), InstructGPT, LoRA, and "
    "Retrieval-Augmented Generation (Lewis et al.). Use this for any question "
    "about what these papers say, their methods, results or terminology. If a "
    "retrieved passage refers to a section or citation you still need, search "
    "again with different wording."
)

# create_retriever_tool defaults document_prompt to "{page_content}", which hands
# the agent bare text with no metadata. The first run showed exactly what that
# costs: the agent cited "pages 6-7" (invented) and "arXiv:1810.04805v2, page 1"
# (scraped from the arXiv stamp printed inside the chunk). It was not lying so
# much as guessing from the only evidence it had. This mirrors format_docs in
# query.py so both paths see the same provenance.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[{source} p.{page_label}]\n{page_content}"
)

# Mirrors the groundedness rules of the chain's prompt so the comparison isolates
# one variable: who decides when to retrieve. Answer quality rules stay equal.
#
# This prompt took three runs, and the failures are worth recording.
#
# Run 1 had no small-talk exception: the agent skipped retrieval for "Hi, how are
# you?" (correct) and then answered "I don't know" (useless). Telling it not to
# search was not the same as telling it what to do instead.
#
# Run 2 added "reply naturally to greetings and small talk" - and the agent
# answered "The capital of Portugal is Lisbon." The exception created a category
# of question it could answer without retrieving, and it filed an out-of-corpus
# factual question into that category. Fixing politeness reopened the exact
# hallucination hole the project exists to close.
#
# So the exception now covers only utterances that ask for no information at all,
# and the ban on answering factual questions from memory is stated separately and
# absolutely. The chain never needed this: it cannot answer from its own
# knowledge because retrieved context is the only text it ever sees. An agent's
# discretion to skip retrieval is the same discretion to answer ungrounded.
SYSTEM_PROMPT = """You answer questions about a corpus of machine learning papers.

Greetings and pleasantries ("hello", "thanks", "how are you") ask for no
information: reply briefly and naturally, and offer to answer questions about
the papers. Do not search for these.

Everything else is a request for information, and you must search for it first -
whatever the topic, and however confident you are of the answer. You have no
other source. If the retrieved passages fully answer the question, answer it. If
they answer it only partly, say what they support and state explicitly what is
missing. If they do not address it, say you don't know.

Never answer a factual question from your own knowledge, even one you are
certain about and even if it seems trivial. A question you can answer without
the papers is still a question the papers must answer.

Cite sources exactly as the retrieved passages label them, copying the document
name and page as given. Never infer or adjust a page number."""


def build_agent(db, k=DEFAULT_K):
    """Wrap the retriever as a tool and hand it to the agent."""
    retriever_tool = create_retriever_tool(
        db.as_retriever(search_kwargs={"k": k}),
        name="search_ml_papers",
        description=TOOL_DESCRIPTION,
        document_prompt=DOCUMENT_PROMPT,
    )
    return create_agent(
        model=ChatOpenAI(model=AGENT_MODEL, temperature=0),
        tools=[retriever_tool],
        system_prompt=SYSTEM_PROMPT,
    )


def run_agent(agent, question):
    """Invoke the agent and record what it actually did.

    The agent returns the whole message history, so the tool calls are visible:
    how many searches it ran and what it searched for. Those queries are worth
    reading - the agent rewrites the user's wording, which is the query-rewriting
    step the eval work suggested, arrived at from a different direction.
    """
    start = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    elapsed = time.perf_counter() - start

    queries = [
        call["args"].get("query", "")
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    return {
        "answer": result["messages"][-1].content,
        "searches": len(queries),
        "queries": queries,
        "seconds": elapsed,
    }


def run_chain(chain, question):
    """Same shape as run_agent. The chain always retrieves exactly once."""
    start = time.perf_counter()
    answer = chain.invoke(question)
    return {
        "answer": answer,
        "searches": 1,  # structural, not a choice
        "queries": [question],  # verbatim - the chain cannot reword
        "seconds": time.perf_counter() - start,
    }


def compare(chain, agent, questions):
    for question in questions:
        print("=" * 72)
        print(f"Q: {question}\n")
        for label, result in (
            ("CHAIN", run_chain(chain, question)),
            ("AGENT", run_agent(agent, question)),
        ):
            print(
                f"--- {label}  ({result['searches']} search(es), {result['seconds']:.1f}s)"
            )
            for query in result["queries"]:
                print(f"    searched: {query!r}")
            print(f"    {result['answer']}\n")


if __name__ == "__main__":
    db = load_vector_store()
    chain = build_chain(db)
    agent = build_agent(db)

    compare(
        chain,
        agent,
        [
            # No retrieval needed. The chain searches anyway and pads the prompt
            # with four irrelevant chunks; the agent should just answer.
            "Hi, how are you?",
            # Squarely in the corpus. Both should retrieve - the question is
            # whether the agent rewords the query before doing so.
            "What rank does the LoRA paper find sufficient for adapting Wq and Wv?",
            # The interesting one. The passage that compares BERT and GPT defers
            # to "the two pre-training tasks presented in Section 3.1" rather than
            # naming them. A single retrieval inherits that gap; an agent can
            # notice the pointer and search again.
            "How does BERT's pre-training objective differ from GPT's, and what "
            "exactly are those two tasks?",
            # Out of corpus. Both must refuse, but the agent has an extra way to
            # fail: it can skip the tool and answer from its own knowledge.
            "What is the capital of Portugal?",
        ],
    )

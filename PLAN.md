# RAG + LangChain Demo — Step-by-Step Plan

_Goal: a small, working retrieve-then-answer pipeline (LangChain + Chroma + OpenAI) you can put on GitHub and talk about in interviews. Stretch: FastAPI endpoint + LangGraph agent._

_Stack decided: **OpenAI API** (LLM + embeddings) · **Chroma** (vector DB, local) · **LangChain 1.x** · public corpus._

_How to use this plan: work phase by phase. Each step has a **checkpoint** — don't move on until it passes. Ask me when stuck, but try 10 minutes yourself first (that struggle is where the learning happens)._

---

## Phase 0 — Project setup (~30 min)

**Learning goal:** a clean, reproducible Python project — this is itself a portfolio signal.

1. **Create the repo structure** inside `rag_demo/`:
   ```
   rag_demo/
   ├── data/           # source documents (PDFs)
   ├── src/
   │   ├── ingest.py   # build the vector store
   │   └── query.py    # ask questions
   ├── .env            # OPENAI_API_KEY=... (never committed)
   ├── .gitignore      # .env, chroma_db/, __pycache__/, .venv/
   ├── requirements.txt
   └── README.md
   ```
2. **Virtual environment:**
   ```bash
   cd rag_demo
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies** (put these in `requirements.txt`, then `pip install -r requirements.txt`):
   ```
   langchain
   langchain-openai
   langchain-chroma
   langchain-community
   pypdf
   python-dotenv
   ```
4. **API key:** create a key at platform.openai.com → put `OPENAI_API_KEY=sk-...` in `.env`. Load it in code with `from dotenv import load_dotenv; load_dotenv()`. Set a low usage limit ($5) in the OpenAI dashboard — the whole project costs well under $1.
5. **Git:** `git init`, commit the skeleton. Confirm `.env` is ignored (`git status` must not show it).

**Checkpoint:** `python -c "from langchain_openai import ChatOpenAI; print(ChatOpenAI(model='gpt-4o-mini').invoke('say ok').content)"` prints a reply.

---

## Phase 1 — Pick the corpus (~30 min)

**Learning goal:** RAG quality starts with the data, not the model.

**Suggested corpus (demos well, safe to publish):** 5–8 seminal ML papers as PDFs from arXiv — e.g. *Attention Is All You Need*, the RAG paper (Lewis et al. 2020), BERT, GPT-3, LoRA, InstructGPT. Why this works: you know the material, interviewers know the material, and "chat with ML papers" is instantly understandable.

1. Download the PDFs into `data/`.
2. Skim each one enough that you can later judge whether answers are correct — you can't evaluate a RAG system over documents you don't know.

**Checkpoint:** `data/` holds 5–8 PDFs; you can name one specific fact from each that a good system should retrieve.

---

## Phase 2 — Ingestion: load → chunk → embed → store (~half a day)

**Learning goal:** the core RAG mechanic. Everything here happens **offline, once** — that's the key mental model (ingestion vs. query time).

Build `src/ingest.py` in four steps:

1. **Load.** `PyPDFLoader` (from `langchain_community.document_loaders`) per file → list of `Document` objects. Print how many pages you got.
2. **Chunk.** `RecursiveCharacterTextSplitter` with `chunk_size=1000, chunk_overlap=150`. *Understand before moving on:* why chunk at all? (Embedding quality degrades on long text; retrieval granularity.) Why overlap? (So facts spanning a boundary survive.) Print 2–3 chunks and read them.
3. **Embed + store.** `OpenAIEmbeddings(model="text-embedding-3-small")` + `Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")`. One call does both: embeds every chunk and writes them to a local Chroma DB.
4. **Sanity-check retrieval without any LLM:**
   ```python
   db.similarity_search("what problem does attention solve?", k=4)
   ```
   Read the 4 returned chunks. Are they from the right paper? This step — retrieval quality inspected by eye — is what separates people who understand RAG from people who copied a tutorial.

**Checkpoint:** `chroma_db/` exists on disk; a similarity search for a fact you planted in Phase 1 returns a chunk containing it.

**Concepts you should now be able to explain:** embedding, vector similarity (cosine), chunking trade-offs, why the vector DB persists.

---

## Phase 3 — Generation: retrieve-then-answer chain (~half a day)

**Learning goal:** composing retrieval + prompt + LLM with LCEL (LangChain's pipe syntax — the current standard; `LLMChain` etc. are deprecated).

Build `src/query.py`:

1. **Reopen the store:** `Chroma(persist_directory="chroma_db", embedding_function=embeddings)` → `retriever = db.as_retriever(search_kwargs={"k": 4})`.
2. **Prompt template** — force groundedness:
   ```
   Answer using ONLY the context below. If the context doesn't contain
   the answer, say you don't know. Cite the source document.

   Context: {context}
   Question: {question}
   ```
3. **Chain (LCEL):**
   ```python
   chain = (
       {"context": retriever | format_docs, "question": RunnablePassthrough()}
       | prompt
       | ChatOpenAI(model="gpt-4o-mini")
       | StrOutputParser()
   )
   ```
   where `format_docs` joins `doc.page_content` (+ `doc.metadata["source"]`) into one string. Trace by hand what flows through each `|` — that's the interview-ready understanding.
4. **Test three question types:** (a) answerable from one chunk, (b) needing chunks from two papers, (c) *not* in the corpus at all. For (c) the system must say "I don't know" — if it hallucinates, tighten the prompt. This is your hallucination-control story for interviews.

**Checkpoint:** correct, cited answers for (a) and (b); honest refusal for (c). **← This is the MVP. Commit and celebrate.**

---

## Phase 4 — Evaluate, document, publish (~half a day)

**Learning goal:** turning a script into a portfolio piece.

1. **Mini eval set:** 8–10 question/expected-answer pairs in `eval_questions.md`. Run them; note failures; fix one (usually by changing `k`, chunk size, or the prompt). Write down what you changed and why — that's your "how did you evaluate it?" answer.
2. **README.md** with: what it does (2 sentences), architecture diagram (ingestion vs. query flow — ASCII is fine), setup instructions, example Q&A output, and a "design decisions" section (chunk size, k, embedding model, groundedness prompt).
3. Push to GitHub under your `data-science-portfolio` account; link it from the CV's projects section.

**Checkpoint:** a stranger could clone the repo and run it from the README alone.

---

## Phase 5 — Stretch A: FastAPI endpoint (~half a day)

_Doubles as your FastAPI homework item._

1. `pip install fastapi uvicorn` (add to requirements).
2. `src/api.py`: a `POST /ask` endpoint taking `{"question": "..."}`, calling the chain, returning `{"answer": ..., "sources": [...]}`. Load the vector store once at startup, not per request.
3. Run `uvicorn src.api:app --reload`, test via the auto-generated Swagger UI at `/docs`.

**Checkpoint:** a curl/Swagger request returns an answer with sources.

---

## Phase 6 — Stretch B: LangGraph agent (~1 day)

**Learning goal:** the shift from a fixed chain to an agent that *decides* when to retrieve — the current direction of the field (LangChain v1's `create_agent` pattern).

1. `pip install langgraph`.
2. Wrap the retriever as a **tool** (`create_retriever_tool`), give it a clear name and description — the description is how the LLM decides to use it.
3. Build the agent with `create_agent` (from `langchain.agents`) with the retriever tool.
4. Compare behaviour vs. the Phase 3 chain: ask "hi, how are you?" — the chain retrieves pointlessly, the agent doesn't. Ask a corpus question — the agent chooses to call the tool. Put this comparison in the README; it shows you understand *why* agents exist, not just how to wire them.

**Checkpoint:** you can show one query where the agent skips retrieval and one where it retrieves, and explain the difference.

---

## Interview cheat-sheet (fill in as you go)

After each phase, write 2–3 sentences answering:

- Why RAG instead of fine-tuning? (fresh/private data, cheaper, citable sources, no retraining)
- How does retrieval actually work? (embeddings → cosine similarity in vector space)
- How did you control hallucination? (groundedness prompt + refusal test)
- How did you evaluate it? (Phase 4 eval set + what you tuned)
- Chain vs. agent — when each? (Phase 6 comparison)

---

## Progress

- [ ] Phase 0 — setup
- [ ] Phase 1 — corpus
- [ ] Phase 2 — ingestion ← the heart of it
- [ ] Phase 3 — generation ← MVP
- [ ] Phase 4 — eval + README + publish
- [ ] Phase 5 — FastAPI
- [ ] Phase 6 — LangGraph agent

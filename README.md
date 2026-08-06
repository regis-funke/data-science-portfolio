# RAG over ML papers

A retrieve-then-answer pipeline over six seminal machine learning papers, built
with LangChain, Chroma and the OpenAI API. Ask a question, get an answer drawn
only from the papers, with the source document cited — or an explicit refusal
when the corpus does not contain the answer.

The interesting part is not the pipeline, which is small. It is the evidence
behind each setting: every configuration below was measured against a
ten-question evaluation set, and two of the decisions reverse what seemed
obvious at the time.

## Architecture

Ingestion runs once, offline. Everything after it is free until you ask a
question.

```
INGESTION (offline, once)                       ingest.py

  data/*.pdf
      |
      v
  PyPDFLoader ................ 215 Documents, one per page
      |
      v
  clean .................. NFKC ligatures, rejoin hyphenated
      |                    line breaks, source -> filename
      v
  RecursiveCharacterTextSplitter ....... 863 chunks, 1000/150
      |
      v
  OpenAIEmbeddings ........... text-embedding-3-large
      |
      v
  Chroma  ->  chroma_db/          (persisted to disk)


QUERY (per question, ~1.5s)                      query.py

  question
      |
      +----------------------------+
      |                            |
      v                            |
  embed + similarity search        | (passed through unchanged)
      |  k=4                       |
      v                            |
  4 chunks -> format_docs          |
  (prefixed with source, page)     |
      |                            |
      +-------------+--------------+
                    |
                    v
            groundedness prompt
                    |
                    v
            gpt-4o-mini, temperature 0
                    |
                    v
            answer + citation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env      # never committed
```

Put PDFs in `data/`, then:

```bash
python src/ingest.py     # build the vector store (~1 min, a few cents)
python src/query.py      # ask the built-in test questions
python src/evaluate.py   # sweep configurations over the eval set
```

`ingest.py` deletes and rebuilds the store on every run, so the index on disk
always matches the settings in the file.

## Example

```
Q: What rank does the LoRA paper find sufficient for adapting Wq and Wv?

The LoRA paper finds that a rank as small as one (r = 1) is sufficient for
adapting both Wq and Wv on the WikiSQL and MultiNLI datasets.
(source: [LORA- LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS.pdf p.10])
```

```
Q: What is the capital of Portugal?

I don't know.
```

The second answer is the point. The model knows the capital of Portugal
perfectly well; the corpus does not contain it, so it declines.

## Design decisions

Measured on a ten-question eval set (`eval_questions.md`) with expected answers
verified against the source PDFs. Full answers per configuration are in
`eval_results/`.

### Embedding model: `text-embedding-3-large`

| config | score |
|---|---|
| 1000/150, k=4, `-small` | 7 / 10 |
| 1500/200, k=4, `-small` | 7 / 10 |
| 1000/150, k=7, `-small` | 8 / 10 |
| **1000/150, k=4, `-large`** | **10 / 10** |

The highest-leverage variable by a distance, and the one that got the least
attention until the sweep. It fixed both failures that chunk size and k could
not, while generating from *less* context than k=7 (3,644 chars vs 6,250) and
costing about two cents more to build.

### Chunk size 1000, overlap 150

600 loses to 1000 even at a matched context budget. The reason is specific and
visible: BERT's Figure 1 caption block is the only place in the corpus that
names its two pre-training objectives, and figure text has no paragraph
structure for the splitter to respect. At 600 characters it fragments into
pieces too sparse to retrieve; at 1000 it survives intact. 1500 tested no better
than 1000.

### k = 4

k=6 returned near-duplicate chunks. k=7 sent 76% more context and scored no
better — it still answered two questions wrong, because the *wrong chunk was
ranked first* in both cases. More context is not better retrieval.

### Plain similarity search, not MMR

Maximal Marginal Relevance was tested and rejected. At `lambda_mult=0.5` it
evicted the chunk that answered the question, replacing it with unrelated
material from two other papers — it rewards dissimilarity from what has already
been picked, which is wrong when the answer lives in one document. At 0.8 it was
byte-identical to plain search. No useful operating point on this corpus. Kept
as an option in `preview()` for comparison.

### Groundedness prompt with three outcomes

Answer fully / answer partially and name what is missing / say you don't know.
An earlier binary version discarded usable information: asked to compare BERT
and GPT, it held a chunk stating GPT's objective and refused outright because
BERT's was missing. The looser wording still refuses out-of-corpus questions —
verified, not assumed.

### PDF text cleaning

Two fixes, both specific to LaTeX-produced PDFs, both applied before chunking:

- **NFKC normalization.** LaTeX emits `fi` as the single character U+FB01, so
  `fine-tuning` is stored as `ﬁne-tuning` and a query for the former will not
  match it. This affected BERT's central concept.
- **Rejoining hyphenated line breaks.** `representa-\ntion` embeds as two
  meaningless fragments. Known limitation: the rule cannot distinguish a soft
  hyphen from a real one, so `task-specific` broken across lines becomes
  `taskspecific`. Accepted — the merged form still embeds near the correct one,
  a severed word embeds near nothing.

## Two conclusions I had to revise

**"The query was the bottleneck, not the index."** A deliberately vague question
— *what problem does attention solve?* — ranked the paper's title and author
block above the passage that answered it, with a 3.2% relative spread across the
top four hits, an ordering barely distinguishable from noise. Rephrasing it in
the paper's own vocabulary fixed it, and I concluded retrieval was healthy.

Re-running the same unrewritten query against the `-large` store ranks the
answering chunk **first**, spread 7.4%, title block absent. The conclusion was
half right: phrasing and embedding quality are two ways of closing the same gap,
and the cheap diagnosis hid a fixable weakness in the index.

**"Filter the boilerplate chunks."** Title blocks and bibliography entries kept
polluting results, and a length or junk-character heuristic would have removed
them easily. It would also have deleted the chunk of Figure 1 label text —
`E[CLS] E1 E[SEP]... NSP Mask LM` — which is the only place the corpus names
BERT's two pre-training tasks. The prose defers to "Section 3.1"; the figure
answers the question. Filter proposed, evidence found against it, filter dropped.

## Known limitations

**Retrieval cannot follow cross-references.** Both BERT and GPT-3 define their
pre-training objectives by pointer — "the two pre-training tasks presented in
Section 3.1", "similar to the process described in [RWC+19]". Retrieval lands on
the right page and the page defers elsewhere. A human follows the pointer; a
fixed chain cannot. This is the clearest argument for an agent that can decide to
retrieve a second time.

**Refusal tests do not prove groundedness.** Both refusal questions passed under
every configuration, yet two configurations asserted that BERT fine-tuning
updates all parameters while retrieving *no BERT chunk at all* — the claim came
from the model's own knowledge and sat next to a citation to a different paper.
A refusal test only catches leakage when the model knows it has nothing. Grading
answers against their retrieved sources is what catches the rest, which is why
the eval harness records them.

**A single query vector skews to one document.** No configuration retrieved from
both papers when asked to compare two. Multi-query retrieval is the obvious next
step and is not implemented.

**Citation fidelity is not enforced.** The model occasionally drops a page number
or invents a range like `p.2-3` by merging two chunks. Returning sources
programmatically from the retrieved documents, rather than trusting the model to
transcribe them, is the robust fix.

## Layout

```
rag_demo/
├── data/                 # source PDFs (gitignored)
├── src/
│   ├── ingest.py         # load -> clean -> chunk -> embed -> store
│   ├── query.py          # retrieval preview + retrieve-then-answer chain
│   └── evaluate.py       # sweep configurations, write a graded report
├── eval_questions.md     # eval set, rubric, results, findings
├── eval_results/         # generated reports, one per sweep
├── chroma_db/            # vector store (gitignored, rebuildable)
└── requirements.txt
```

## Corpus

Attention Is All You Need · BERT · Language Models are Few-Shot Learners (GPT-3)
· Training Language Models to Follow Instructions (InstructGPT) · LoRA ·
Retrieval-Augmented Generation (Lewis et al.)

Chosen because the material is widely known: an interviewer can judge whether an
answer is correct without reading the source, and I can judge whether retrieval
found the right passage.

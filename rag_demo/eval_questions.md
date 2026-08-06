# Evaluation set

Ten questions used to compare retrieval and prompt configurations. Every
expected answer below was verified against the source PDF, not from memory.

The set is deliberately unbalanced: easy questions confirm nothing is broken,
but only the hard ones discriminate between configurations. An earlier
three-question set was useless for tuning because two of the three passed under
every setting tested.

**Corpus:** Attention Is All You Need · BERT · GPT-3 (Language Models are
Few-Shot Learners) · InstructGPT · LoRA · RAG (Lewis et al.)

## Scoring

| Grade | Meaning |
|---|---|
| **pass** | Contains the expected fact, cites the right document |
| **partial** | Correct as far as it goes, but omits part of the expected answer, or states the fact without a usable citation |
| **fail** | Wrong, hallucinated, or refuses when the corpus does contain the answer |

For Q9 and Q10 a refusal *is* the pass. Answering them is the failure — that is
the hallucination test.

Record the grade plus the answer text for each configuration, so regressions are
visible and not just aggregate scores.

---

## Q1 — single chunk, factual

**Question:** What is the maximum path length in a self-attention layer?

**Expected:** O(1) — constant. A self-attention layer connects all positions
with a constant number of sequentially executed operations, whereas a recurrent
layer requires O(n).

**Source:** attention-is-all-you-need.pdf, Table 1 / §4

**Tests:** basic retrieval and citation. Passed under every configuration tried
so far, so it functions as a smoke test rather than a discriminator.

---

## Q2 — single chunk, reasoning stated in the text

**Question:** Why are the dot products scaled by 1/√dk in scaled dot-product
attention?

**Expected:** For large dk the dot products grow large in magnitude, pushing the
softmax into regions where it has extremely small gradients. Scaling counteracts
this.

**Source:** attention-is-all-you-need.pdf §3.2.1

**Tests:** whether the retrieved chunk carries the *justification* and not only
the formula. Notation-dense passages tokenize badly, so this is a fair test of
the embedding on mathematical text.

---

## Q3 — figure and table text

**Question:** What are BERT's two pre-training tasks?

**Expected:** Masked language modelling (Mask LM) and next sentence prediction
(NSP).

**Source:** BERT.pdf §3.1, and the Figure 1 caption block

**Tests:** the known weak point. In the body the paper defers to "the two
pre-training tasks presented in Section 3.1"; the names appear in Figure 1's
label text, which has no sentence structure. This question is what showed
chunk_size 600 losing to 1000 — dense figure text fragments at small chunk
sizes. **Most valuable question in the set.**

---

## Q4 — precise numeric detail

**Question:** What proportion of tokens does BERT mask during pre-training?

**Expected:** 15% of all WordPiece tokens in each sequence, at random.

**Source:** BERT.pdf §3.1

**Tests:** numeric precision. Watch for the model inventing a plausible
percentage, and for the 80/10/10 split of mask/random/unchanged being confused
with the 15%.

---

## Q5 — cross-reference trap

**Question:** How does BERT's pre-training objective differ from GPT's?

**Expected:** BERT is bidirectional and uses two pre-training tasks (MLM, NSP);
GPT trains a left-to-right Transformer LM. A complete answer names both tasks. A
partial answer notes the bidirectional/unidirectional split without naming them.

**Source:** BERT.pdf §A.4 and Figure 1

**Tests:** handling of context that points elsewhere rather than stating the
fact. Note the question says GPT, not GPT-3 — an earlier version said GPT-3 and
the model correctly refused, since the BERT paper compares against GPT-1.

---

## Q6 — cross-document comparison

**Question:** Which approach freezes the pre-trained model weights during
adaptation, LoRA or BERT fine-tuning?

**Expected:** LoRA freezes the pre-trained weights and trains injected low-rank
decomposition matrices; BERT fine-tunes all parameters.

**Source:** LORA.pdf §1 and BERT.pdf §3.2

**Tests:** whether a k-chunk window can hold evidence from two documents at
once. Watch whether both are cited or only one.

---

## Q7 — quantitative claim, abstract

**Question:** By how much does LoRA reduce trainable parameters and GPU memory
compared with fine-tuning GPT-3 175B with Adam?

**Expected:** 10,000× fewer trainable parameters and 3× less GPU memory.

**Source:** LORA.pdf, abstract

**Tests:** retrieval of two figures from one sentence. The paper also mentions a
separate 10,000× checkpoint-size reduction (350GB → 35MB) — conflating the two
is a partial, not a pass.

---

## Q8 — counter-intuitive finding

**Question:** What rank does the LoRA paper find sufficient for adapting Wq and
Wv?

**Expected:** A rank as small as one suffices for adapting both Wq and Wv on
these datasets; adapting Wq alone needs a larger r.

**Source:** LORA.pdf §7.2, Table 6

**Tests:** retrieval from text surrounding a results table. The caveat about Wq
alone is the part most likely to be dropped.

---

## Q9 — plausible but absent (refusal test)

**Question:** How does chain-of-thought prompting improve reasoning performance?

**Expected:** **A refusal.** The corpus does not cover chain-of-thought
prompting — it postdates these papers. The model knows the answer from its own
training, so answering means the groundedness prompt has failed.

**Source:** none

**Tests:** the hard hallucination case. Unlike Q10 this question sits squarely
in the corpus's domain, so nothing about it *feels* out of scope.

---

## Q10 — clearly out of scope (refusal test)

**Question:** What is the capital of Portugal?

**Expected:** **A refusal.**

**Source:** none

**Tests:** the easy hallucination case. Passed under every configuration so far.
Kept as a regression check on prompt changes — when the prompt was loosened to
allow partial answers, this question is what confirmed refusals still worked.

---

## Results

### Sweep, 2026-08-06 (`eval_results/eval_2026-08-06_1925.md`)

All four configurations used gpt-4o-mini at temperature 0.

| Q | base-1000-k4 | large-1500-k4 | base-1000-k7 | emb-large-1000-k4 |
|---|---|---|---|---|
| Q1 | pass | pass | pass | pass |
| Q2 | pass | pass | pass | pass |
| Q3 | fail | partial | fail | **pass** |
| Q4 | fail | pass | fail | **pass** |
| Q5 | pass | partial | pass | pass |
| Q6 | partial | partial | pass | pass |
| Q7 | pass | pass | pass | pass |
| Q8 | pass | pass | pass | pass |
| Q9 | pass | pass | pass | pass |
| Q10 | pass | pass | pass | pass |
| **pass / partial / fail** | 7 / 1 / 2 | 7 / 3 / 0 | 8 / 0 / 2 | **10 / 0 / 0** |

| config | retrieval | generation | context chars |
|---|---|---|---|
| base-1000-k4 | 0.18s | 1.46s | 3,535 |
| large-1500-k4 | 0.18s | 1.30s | 5,034 |
| base-1000-k7 | 0.23s | 1.52s | 6,250 |
| emb-large-1000-k4 | 0.23s | 1.30s | 3,644 |

**Adopted: 1000/150, k=4, text-embedding-3-large.**

### What the sweep showed

**The embedding model mattered more than anything else tested.** Switching from
`text-embedding-3-small` to `-large` at identical chunk size and k took the score
from 7/10 to 10/10, fixing both failures that chunk size and k could not. It is
also the cheapest configuration to generate with — fewer context characters than
k=7 and the fastest mean generation time. Chunk size and k had absorbed far more
tuning effort for far less return.

**More context is not better retrieval.** At k=7 the model still failed Q3 and
Q4. On Q4 it answered with BERT's 80/10/10 mask-replacement split rather than the
15% masking rate under both k=4 and k=7, because the wrong passage ranked first
in each. Seven chunks did not help; better ranking did.

**Refusal tests do not prove groundedness.** Q9 and Q10 passed under every
configuration, yet two configurations leaked unsourced facts elsewhere:

- Q6, base-1000-k4 and large-1500-k4: both asserted that BERT fine-tuning updates
  all parameters while retrieving **no BERT chunk at all**. The claim came from
  the model's own knowledge and sat next to a citation to a different paper.
- Q3, large-1500-k4: named MLM and NSP while stating in the same answer that the
  context "does not explicitly name them."

A refusal test only catches leakage when the model knows it has nothing. It
misses the more dangerous case: topically adjacent context, with the gap quietly
filled from pre-training. Grading answers against the retrieved sources — which
is why the harness records them — is what catches this.

**No configuration retrieved from both documents on Q6.** A single query vector
naming two topics lands nearest one of them. This is the concrete argument for
multi-query retrieval, still untested.

**Better embeddings reduced sensitivity to question phrasing.** During Phase 2 a
deliberately vague query — "what problem does attention solve?" — ranked the
paper's title and author block above the passage that answered it, with a 3.2%
relative spread across the top four hits: an ordering barely distinguishable
from noise. Rephrasing the question in the paper's own vocabulary fixed it, and
the conclusion at the time was that the query, not the index, was at fault.

Re-running the same unrewritten query against the `-large` store ranks the
answering chunk **first**, with the title block absent from the top four and the
relative spread at 7.4%. So the original conclusion was only half right. Query
phrasing and embedding quality are two ways of closing the same gap, and the
cheap diagnosis — "your question was vague" — hid a fixable weakness in the
index. Worth remembering as a bias: the variable already under examination is
not necessarily the one that matters.

(Spreads are quoted relative to the top score. Two embedding models occupy
different vector spaces, so their absolute distances cannot be compared.)

### Still untested

- Multi-query / HyDE query rewriting (motivated by Q6; the phrasing argument for
  it is weaker now that `-large` handles vague queries)
- Filtering bibliography chunks — deliberately *not* filtering figure captions,
  which supplied the Q3 answer
- A larger LLM; every run above used gpt-4o-mini

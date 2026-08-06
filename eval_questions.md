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

Record one table per configuration.

### 1000/150, k=4, text-embedding-3-small, gpt-4o-mini

| Q | Grade | Note |
|---|---|---|
| Q1 | pass | |
| Q2 | | |
| Q3 | pass | names MLM and NSP |
| Q4 | | |
| Q5 | pass | names both tasks, states remaining gap |
| Q6 | | |
| Q7 | | |
| Q8 | | |
| Q9 | | |
| Q10 | pass | refuses |

Blank rows are not yet run — Q1, Q3, Q5 and Q10 are carried over from the
earlier three-question set.

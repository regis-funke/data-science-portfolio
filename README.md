# Data Science Portfolio — Regis Funke

MSc Data Science. I work mostly in Python — retrieval and LLM pipelines, PyTorch,
and applied evaluation. Open to GenAI and LLM engineering roles.

**This repository is the index and the archive.** Current work has its own repo
and is linked first below. What remains here is the M.Sc. and thesis-era work,
2022–2023: three projects worth reading in full, two R analyses that open in the
browser without installing anything, and the coursework underneath for range.

Every folder has its own README, and all of them render as pages at
[regis-funke.github.io/data-science-portfolio](https://regis-funke.github.io/data-science-portfolio/).

---

## Current work — in its own repository

### [→ rag-eval-lab](https://github.com/regis-funke/rag-eval-lab) · 2026

**RAG over ML papers, with every setting decided by measurement.** A
retrieve-then-answer pipeline over six seminal ML papers — LangChain, LangGraph,
Chroma, FastAPI, OpenAI. Answers cite their source document and page, or refuse
when the corpus does not contain the answer.

Swapping the embedding model from `text-embedding-3-small` to `-large` took the
eval score from 7/10 to 10/10 — a larger gain than chunk size or retrieval depth,
both of which had absorbed far more tuning effort. Maximal Marginal Relevance
was tested and rejected. Two conclusions that later evidence overturned are
documented rather than quietly corrected.

**Python · LangChain · LangGraph · Chroma · FastAPI · OpenAI API**

---

## Featured — the three worth reading in full

### [ViT fine-tuning strategies for object detection](Vision%20Transformers/object_detection) · 2023

Three fine-tuning strategies for a Vision Transformer on the same detection
task, compared on generalised IoU:

| Strategy | Average GIoU |
|---|---|
| Custom head only | 0.830 |
| Custom head + 2 transformer layers | 0.845 |
| Full ViT relearning | 0.858 |

Localisation improves monotonically with the number of unfrozen parameters, and
the gap between freezing everything and retraining everything is under three
GIoU points — which is the interesting part, given the difference in training
cost.

**Python · PyTorch · PyTorch Lightning · Transformers**

### [OCR engine comparison](2_Semester/application_project) · 2023

Which OCR engine holds up on real photographs of food packaging? 85 images from
openfoodfacts.org, ground truth built with Abbyy FineReader and corrected by
hand, accuracy scored by Levenshtein ratio against that ground truth.

EasyOCR produced the largest share of results above 0.9 similarity. Pytesseract
was faster but less accurate. docTR degraded fastest as image quality dropped.

The methodology is the point: a benchmark is only as good as its ground truth,
so most of the work went into building one.

**Python · Pytesseract · EasyOCR · python-doctr · OpenCV · python-Levenshtein**

### [Credit default prediction](2_Semester/deep_learning/DL_Portfolio_Exam_2) · 2023

A neural network for credit default on the South German Credit dataset (1,000
anonymised customers), extending an earlier classical-ML treatment of the same
problem. Best balanced validation accuracy 0.766, test accuracy around 70%.

The network beat the previous best model — an SVM — but the honest conclusion
is that the gain in balanced accuracy was marginal, and does not obviously
justify the added complexity on a dataset this size. Most of the work went into
the parts that decide whether a credit model is usable at all: handling class
imbalance through weight adjustment, architecture search, and hyperparameter
tuning with `ray.tune`.

**Python · PyTorch · ray.tune · Pandas · Plotly**

---

## Read in the browser — data visualisation in R

Both are knitted to HTML and served through GitHub Pages, so the Plotly charts
stay interactive and nothing needs to be installed to look at them.

### [Well-being against ecological footprint: the Happy Planet Index](https://regis-funke.github.io/data-science-portfolio/1_Semester/data_viz/DV_Portfolio_Exam_1/RF_challenge.html) · 2022

The Happy Planet Index scores countries on well-being and sustainability rather
than output, which makes it a good test of whether a visualisation can hold four
variables at once without becoming unreadable. Four views: the relationships
between HPI variables, Zimbabwe's trajectory over time, a world map under a
chosen projection, and the distribution per continent.

[Source](1_Semester/data_viz/DV_Portfolio_Exam_1) · **R · tidyverse · ggplot2 · Plotly · sf · rnaturalearth · GGally**

### [German inflation and consumer spending since 1991](https://regis-funke.github.io/data-science-portfolio/1_Semester/data_viz/DV_Portfolio_Exam_2/DV_Exam_2.html) · 2022

Consumer price index since 1992, month-by-month inflation rates, and how price
development differs across spending categories — some rising steadily, others
cyclical. The interesting comparison is between categories like transport and
education, because the aggregate rate hides how differently it lands depending
on what a household actually spends money on.

Mixed toolchain by design: the DESTATIS XML and XLSX exports are prepared in a
Python notebook, then visualised in R.

[Source](1_Semester/data_viz/DV_Portfolio_Exam_2) · **R · tidyverse · Plotly · Python (pandas)**

---

## Also here

- **[ViT image classification](Vision%20Transformers/object_identification)** —
  Vision Transformers applied to Caltech 101, CIFAR-10, a snacks dataset and
  dementia severity classification.
- **[DPT-DinoV2 depth estimation](Dinov2)** — fine-tuning DPT-DinoV2-Small from
  KITTI to the NYU Depth dataset, with selective layer unfreezing.
- **[Social media analytics](2_Semester/social_media_analytics)** — an
  end-to-end pipeline: scraping German parliament press releases, filtering with
  LDA, then topic modelling over time. Also word embeddings (Word2Vec,
  FastText) and a content-based article recommender.

## MSc coursework archive, 2023

**Second semester** — [data mining](2_Semester/data_mining) (APRIORI and
FP-Growth) · [deep learning](2_Semester/deep_learning) ·
[social media analytics](2_Semester/social_media_analytics) ·
[application project](2_Semester/application_project)

**First semester** — [data visualisation](1_Semester/data_viz) (Happy Planet
Index in R; inflation and consumer spending) ·
[machine learning](1_Semester/machine_learning) (kNN, random forests, credit
compliance) ·
[multivariate statistics](1_Semester/maths_multivariate_statistics) ·
[tools and programming](1_Semester/tools_and_programming)

---

## Running the code

Almost everything here is a Jupyter notebook or an R Markdown file, committed
with its outputs, so it can be read on GitHub without running anything. That is
deliberate — the outputs are what make a notebook readable.

To execute them:

```bash
git clone https://github.com/regis-funke/data-science-portfolio.git
```

Then install that project's dependencies and run its notebook. There is no
repository-wide environment; each project names what it needs.

The current LLM work is a proper Python package with a pinned
`requirements.txt`, a FastAPI service and its own setup instructions, in
[rag-eval-lab](https://github.com/regis-funke/rag-eval-lab).

## Licence

MIT — see [LICENSE](LICENSE).

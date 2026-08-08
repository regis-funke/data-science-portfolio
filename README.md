# Data Science Portfolio — Regis Funke

MSc Data Science. I work mostly in Python — retrieval and LLM pipelines, PyTorch,
and applied evaluation. Open to data roles.

The four projects below are the ones worth your time; the coursework archive
underneath shows the range. Each project folder has its own README.

---

## Featured

### [RAG over ML papers](rag_demo) · 2026

A retrieve-then-answer pipeline over six seminal ML papers — LangChain, Chroma,
OpenAI. Answers cite their source document, or refuse when the corpus does not
contain the answer.

Every setting was chosen by measurement against a ten-question evaluation set
rather than by default. Swapping the embedding model from
`text-embedding-3-small` to `-large` took the score from 7/10 to 10/10 — a
larger gain than chunk size or retrieval depth, both of which had absorbed far
more tuning effort. Maximal Marginal Relevance was tested and rejected: it
evicted the chunk that answered the question.

The README documents two conclusions that later evidence overturned, and the
limitations the eval exposed — including that passing refusal tests does not
prove groundedness.

**Python · LangChain · Chroma · OpenAI API · pypdf**

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

## MSc coursework, 2023

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

Most projects are Jupyter notebooks and can be read directly on GitHub without
running anything. To execute them:

```bash
git clone https://github.com/regis-funke/data-science-portfolio.git
```

Then install that project's dependencies and run its notebook. `rag_demo` is the
exception — it is a Python package with a pinned `requirements.txt` and its own
setup instructions.

## Licence

MIT — see [LICENSE](LICENSE).

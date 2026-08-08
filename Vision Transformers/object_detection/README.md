# ViT fine-tuning strategies for object detection

Three fine-tuning strategies for a Vision Transformer on the same object
detection task, held otherwise identical: same base model
(`google/vit-base-patch16-224-in21k`), same dataset (Caltech 101), same loss and
metric (Generalised IoU). The only variable is how much of the backbone is
allowed to learn.

## Results

| Strategy | Trainable | Average GIoU |
|---|---|---|
| [Custom head only](pytorch_object_detection_custom_head.ipynb) | detection head | 0.830 |
| [Custom head + last 2 transformer layers](pytorch_object_detection_custom_head_two_transformer_layers.ipynb) | head + 2 layers | 0.845 |
| [Full relearning](pytorch_object_detection_full_vit.ipynb) | entire network | 0.858 |

Localisation improves monotonically with the number of unfrozen parameters, as
expected. The interesting part is the size of the gap: **2.8 GIoU points
separate training a head on a frozen backbone from retraining the whole
network**, for a large difference in training cost. On this dataset, the frozen
backbone already produces features good enough that most of the remaining error
is not fixable by fine-tuning it.

Best and worst cases per run are in the notebooks; maximum GIoU reaches 0.98 for
full relearning, and the minimum is near zero in all three, meaning every
strategy still misses completely on some images.

## Method

- **Base model**: `google/vit-base-patch16-224-in21k`, pretrained on ImageNet-21k
- **Dataset**: Caltech 101, single-object bounding box regression
- **Loss and metric**: Generalised IoU — chosen over plain IoU because it stays
  informative when boxes do not overlap at all, where IoU is flat at zero and
  gives no gradient
- **Stack**: PyTorch, PyTorch Lightning, Transformers, PIL, Matplotlib

---

[← Portfolio index](../../README.md)

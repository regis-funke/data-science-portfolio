# Fine-Tuning DPT-DinoV2 on NYU Depth Dataset

Welcome to my GitHub repository where I share my exploratory journey in fine-tuning the DPT-DinoV2-Small-KITTI model using the NYU Depth Dataset. This project aims to deepen my understanding of Transformer models and their application in depth estimation tasks.

## Overview

In this project, I focused on adapting and fine-tuning the dpt-dinov2-small-kitti model with the dpt-dinov2-small-nyu dataset. The NYU Depth Dataset poses unique challenges for depth estimation, making it an ideal candidate for this exploration. The project encapsulates a comprehensive workflow, from data preprocessing to model training.

## Project Structure

The Jupyter Notebook, which is the centerpiece of this repository, is meticulously structured to provide insights at every step of the process. It includes:

- **Standard and Third-Party Library Imports**: Essential libraries and modules required for the project.
- **PyTorch and PyTorch Lightning Modules**: Utilizing these frameworks for efficient model training and management.
- **Custom Dataset and DataLoader**: Crafting a dataset tailored for depth estimation and setting up DataLoaders for batch processing.
- **Model Definition and Fine-Tuning**: Initializing the DPT-DinoV2 model and selectively unfreezing layers for fine-tuning.
- **Training Process**: Leveraging PyTorch Lightning's Trainer to facilitate the training with early stopping callbacks.

## Key Highlights

- **Model Adaptation**: The project demonstrates how to adapt a pre-trained model to a different dataset and task.
- **Layer Unfreezing**: Strategic unfreezing of the last two layers of the Transformer and the head for fine-tuning.
- **Custom Dataset Handling**: Creation of a custom dataset class for processing depth maps to match the model's output size.
- **In-depth Annotations**: Each code block is accompanied by detailed comments and explanations to elucidate the workflow.

## Conclusion

This project not only aimed at achieving effective fine-tuning of the DPT model but also served as an insightful exploration into Transformer models. I invite fellow enthusiasts and researchers to explore, critique, and build upon this work.

---

[← Portfolio index](../README.md)

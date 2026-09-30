# TG-CMA

This repository contains code to reproduce results from the paper:

rajectory-Guided Cross-Modal Adversarial Attacks with Neighborhood Gradient Estimation.

## Requirements

### Install dependencies

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.28.1
pip install timm==0.6.13
pip install ftfy==6.1.1
pip install ruamel_yaml==0.15.80
```

### Prepare datasets and models

Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in`./configs/Retrieval_flickr.yaml, image_root`.
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).

## Transferability Evaluation

### Image-Text Retrieval Attack Evaluation

Ours(TG-CMA):

```bash
python run_eval.py

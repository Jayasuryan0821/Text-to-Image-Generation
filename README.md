# Baseline Implementation for Stable Diffusion Fine-Tuning

## Overview

This repository contains a baseline implementation for fine-tuning **Stable Diffusion** on the MidJourney Dataset using PyTorch and Hugging Face's `diffusers`,`transformers` library and **Meta's Segment Anything Model** . The implementation runs in a Kaggle environment with **T4x2 GPU** and leverages `accelerate`, `bitsandbytes`, `PEFT`, and `xFormers` for efficient model training.

## Features
- Used **Hugging Face diffusers** for Stable Diffusion fine-tuning on the MidJourney Dataset which consists of images and captions.
- Supports **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning and which is effective for training on GPU's that have low VRAM.
- Implements dataset handling using **Hugging Face datasets**.
- Utilizes **accelerate** for efficient training across multi-GPU's.
- Integrates **DDDIM schedulers** for diffusion model optimization.
- Used **CLIP Score** for evaluating the text-to-image generation of the fine tuned Stable Diffusion model.
- **SAM** was used generate to segmentation masks for the generated images by Stable Diffusion.
 
## Setup & Installation

Ensure you have the required dependencies installed before running the notebook.

### Install Dependencies

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers -U opencv-python pandas numpy matplotlib
pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
pip install -q jupyter_bbox_widget supervision==0.23.0
pip install accelerate datasets transformers
pip install bitsandbytes
pip install peft safetensors wandb
pip install supervision

```

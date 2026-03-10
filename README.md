# 🦙 Fine-Tuning Llama 2 on an Instruction Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LWhvx79nA_AhRQT6JyvDXcmNg4Aojg22#scrollTo=uCaPx3dbsJM1)

Fine-tune **Meta's Llama 2-7B Chat** model on a custom instruction dataset using **QLoRA** (4-bit quantization + LoRA), making it feasible on free-tier Google Colab GPUs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Why QLoRA?](#why-qlora)
- [Prompt Template](#prompt-template)
- [Dataset](#dataset)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Requirements](#requirements)

---

## Overview

This project demonstrates how to fine-tune `meta-llama/Llama-2-7b-chat-hf` on an instruction-following dataset using:

- **QLoRA** — Quantized Low-Rank Adaptation for memory-efficient fine-tuning
- **4-bit NF4 Quantization** via `bitsandbytes`
- **SFTTrainer** from the `trl` library for supervised fine-tuning
- **PEFT** (Parameter Efficient Fine-Tuning) to reduce trainable parameters

The resulting fine-tuned model is saved as `llama-2-7b-chat-finetune`.

---

## Why QLoRA?

Free Google Colab provides ~15GB of VRAM — barely enough to load Llama 2-7B's weights. Full fine-tuning is out of the question due to additional overhead from:

- Optimizer states
- Gradients
- Forward activations

**QLoRA** solves this by:
1. Loading the model in **4-bit precision** (NF4 type)
2. Adding **low-rank adapter layers** (LoRA) — only these are trained
3. Drastically reducing VRAM usage while preserving model quality

---

## Prompt Template

Llama 2 Chat models use the following prompt format:

```
<s>[INST] <<SYS>>
{System Prompt}
<</SYS>>

{User Prompt}
[/INST]

{Model Answer}
</s>
```

> **Note:** A specific prompt template is only required when using the **chat** version of Llama 2. If you use the base model instead, no template is needed.

---

## Dataset

| Dataset | Description | Link |
|---|---|---|
| Original | OpenAssistant Guanaco | [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) |
| Reformatted (1k samples) | Llama 2 template format | [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) |
| Reformatted (full) | Llama 2 template format | [mlabonne/guanaco-llama2](https://huggingface.co/datasets/mlabonne/guanaco-llama2) |

This project trains on the **1k sample** version for speed and resource efficiency.

> How the dataset was reformatted: [Reformatting Notebook](https://colab.research.google.com/drive/11oUe11VosVhHjXIWzRoXSuZJ-3297xZY)

---

## Installation

```bash
pip install -q accelerate peft bitsandbytes transformers trl
```

### HuggingFace Authentication

You'll need a HuggingFace token with access to `meta-llama/Llama-2-7b-chat-hf`:

```python
from google.colab import userdata
from huggingface_hub import login

token = userdata.get('HUGGINGFACEHUB_API_TOKEN')
login(token)
```

---

## Configuration

### Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `lora_r` | 64 | LoRA attention dimension (rank) |
| `lora_alpha` | 16 | LoRA scaling parameter |
| `lora_dropout` | 0.1 | Dropout probability for LoRA layers |
| `bnb_4bit_quant_type` | `nf4` | Quantization type |
| `bnb_4bit_compute_dtype` | `float16` | Compute dtype |
| `learning_rate` | 2e-4 | Initial learning rate (AdamW) |
| `num_train_epochs` | 1 | Training epochs |
| `per_device_train_batch_size` | 1 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Steps before gradient update |
| `optim` | `paged_adamw_32bit` | Optimizer |
| `lr_scheduler_type` | `cosine` | Learning rate schedule |
| `max_grad_norm` | 0.3 | Gradient clipping |

---

## Training

The training pipeline follows these steps:

1. **Load Dataset** — Load and preprocess `mlabonne/guanaco-llama2-1k`
2. **Configure bitsandbytes** — Set up 4-bit NF4 quantization
3. **Load Base Model** — Load `meta-llama/Llama-2-7b-chat-hf` in 4-bit precision
4. **Load Tokenizer** — Configure the LLaMA tokenizer with right-side padding
5. **Configure QLoRA** — Set up LoRA adapters via `LoraConfig`
6. **Set Training Arguments** — Configure `TrainingArguments` with all hyperparameters
7. **Launch SFTTrainer** — Start supervised fine-tuning

```python
# Train Model
trainer.train()
```

Training logs are saved to TensorBoard under `./results`.

---

## Requirements

- Python 3.8+
- CUDA-compatible GPU (≥15GB VRAM recommended)
- HuggingFace account with Llama 2 access approved
- Google Colab (free tier works with QLoRA)

### Libraries

```
accelerate
peft
bitsandbytes
transformers
trl
torch
datasets
```

---

## 📎 References

- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer)

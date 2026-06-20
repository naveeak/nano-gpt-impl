# Shakespeare GPT

A small GPT-style character-level language model trained on Tiny Shakespeare, following Karpathy's "Let's build GPT" tutorial.

## Trained weights
Model weights (~133MB) are hosted on Hugging Face Hub, not in this repo:
👉 https://huggingface.co/naveeak/nano-gpt/tree/main

## Usage

```bash
pip install -r requirements.txt
python generate.py
```

This downloads the checkpoint from HF Hub automatically and generates sample text.

## Training

```bash
python train.py
```

## Architecture
- 6-layer, 6-head transformer, 384 embedding dim
- Character-level tokenization
- Block size 256, trained for 5000 iters

## Repo structure
| File | Purpose |
|---|---|
| `model.py` | Model architecture (attention, blocks, feedforward) |
| `data.py` | Data loading and tokenization |
| `train.py` | Training loop |
| `generate.py` | Load trained model + sample |
| `config.py` | Hyperparameters |

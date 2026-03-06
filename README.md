# 🤖 Billi AI

A tiny language model built completely from scratch using PyTorch. No pre-trained weights, no shortcuts — just pure Python and no math i hate math

---

## What is Billi AI?

Billi AI is a "GPT-style" transformer language model trained on conversation data. It was built from the ground up to understand how large language models actually work under the hood.

- **~800K parameters**
- Character-level tokenizer
- 4-layer transformer with multi-head self-attention
- Trained on conversational data

---

## Project Structure

```
billiai.py              ← model architecture + training + generation
app.py                  ← Gradio web UI
billiai.pt              ← trained model weights
billiai_tokenizer.pkl   ← tokenizer
requirements.txt        ← dependencies
```

---

## Getting Started

### Install dependencies
```bash
pip install torch gradio
```

### Train the model
```bash
python billiai.py --train
```

### Train on your own text
```bash
python billiai.py --train --file yourtext.txt
```

### Generate text
```bash
python billiai.py --generate --prompt "Human: hi\nBilli AI:"
```

### Run the chat UI locally
```bash
python app.py
```

---

## Model Architecture

| Component | Details |
|---|---|
| Type | GPT-style Transformer |
| Parameters | ~800K |
| Layers | 4 |
| Attention heads | 4 |
| Embedding size | 128 |
| Context length | 128 tokens |
| Tokenizer | Character-level |

---

## Training Config

| Setting | Value |
|---|---|
| Steps | 3000 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |

this toook wayy to looong i trained it on a cpuu

---

## Requirements

- Python 3.8+
- PyTorch
- Gradio (for web UI only) kinda cool ngl

---

## Built With

- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)

---

## License

MIT License — free to use, modify and share.

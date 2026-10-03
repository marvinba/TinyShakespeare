# Tiny Shakespeare Transformer

A decoder-only Transformer language model built from scratch in PyTorch and trained on the Tiny Shakespeare dataset.

## Overview

I built this project to deepen my understanding of how modern language models work from the ground up. Rather than relying on high-level Transformer libraries, I implemented the core components myself, including a custom byte-level BPE tokenizer, causal self-attention, multi-head attention, Transformer blocks, training, evaluation, checkpointing, and autoregressive text generation.

The project was inspired by Andrej Karpathy's *Neural Networks: Zero to Hero* series. While the series demonstrates building a character-level language model, I extended the approach by implementing my own byte-level Byte Pair Encoding (BPE) tokenizer and experimenting with Transformer architecture and training choices.

The model was developed and trained using PyTorch in Google Colab with GPU acceleration.

## Features

- Custom byte-level BPE tokenizer built from scratch
- Decoder-only Transformer implemented in PyTorch
- Multi-head causal self-attention
- Token and positional embeddings
- Pre-LayerNorm Transformer blocks
- Residual connections
- GELU feed-forward networks
- Dropout and label smoothing
- AdamW optimizer with weight decay
- Learning-rate warmup followed by cosine decay
- Gradient clipping
- Validation-based model checkpointing
- Autoregressive text generation
- Saved model and tokenizer for inference

## Dataset

The model is trained on the Tiny Shakespeare dataset.

During preprocessing, character names that introduce dialogue are replaced with a special `<Dialogue>` token. This gives the model a consistent representation of dialogue boundaries and provides a starting and stopping point during text generation.

The dataset is split into:

- 90% training data
- 10% validation data

The tokenizer is trained exclusively on the training split.

## Byte-Level BPE Tokenizer

Instead of using character-level tokenization, I implemented a byte-level Byte Pair Encoding tokenizer from scratch.

The tokenizer begins with the 256 possible byte values and a dedicated `<Dialogue>` token. It repeatedly identifies the most frequent adjacent token pair and merges it into a new token until the target vocabulary size is reached.

The final vocabulary contains 1,000 tokens.

The learned BPE merge rules and ID-to-token mappings are saved to:

`checkpoints/tokenizer.json`

This allows the trained tokenizer to be loaded and reused without learning the BPE merges again.

## Model Architecture

The model is a decoder-only Transformer with the following final configuration:

| Hyperparameter | Value |
|---|---:|
| Embedding dimension | 320 |
| Attention heads | 8 |
| Transformer layers | 4 |
| Context length | 256 |
| Vocabulary size | 1,000 |
| Feed-forward expansion | 4x |
| Activation | GELU |
| Dropout | 0.40 |

Each Transformer block uses a pre-LayerNorm architecture:

1. LayerNorm → Multi-Head Causal Self-Attention → Residual Connection
2. LayerNorm → Feed-Forward Network → Residual Connection

A final LayerNorm and linear language-model head produce logits over the vocabulary.

## Training

The final training configuration uses:

| Hyperparameter | Value |
|---|---:|
| Batch size | 64 |
| Training iterations | 5,000 |
| Initial learning rate | 3e-4 |
| Optimizer | AdamW |
| Weight decay | 0.05 |
| Warmup iterations | 250 |
| Learning-rate schedule | Linear warmup + cosine decay |
| Gradient clipping | 1.0 |
| Label smoothing | 0.10 |
| Evaluation interval | 500 |
| Evaluation batches | 200 |

During training, validation loss is periodically evaluated and the model with the lowest validation loss is saved to:

`checkpoints/best_model.pt`

## Model Depth Experiment

I experimented with Transformer depth to determine whether increasing model capacity meaningfully improved validation performance.

| Layers | Validation Loss |
|--------|-----------------|
| 2 | ~3.80 |
| 4 | ~3.72 |
| 6 | ~3.72 |

Three final runs of the 4-layer configuration produced validation losses of:

- 3.7237
- 3.7150
- 3.7190

**Mean validation loss: 3.7192**

The 6-layer model produced nearly identical validation performance. I therefore selected the 4-layer model as the final architecture because it achieved comparable validation loss with fewer layers and lower computational cost.

## Text Generation

Generation begins with the `<Dialogue>` token and proceeds autoregressively.

At each step, the model:

1. Processes the current sequence
2. Predicts logits for the next token
3. Converts the logits into probabilities using softmax
4. Samples the next token
5. Appends it to the sequence

Generation ends when another `<Dialogue>` token is produced or the maximum context length is reached.

The generated token IDs are converted back into bytes using the saved tokenizer and decoded into UTF-8 text.

## Example Output

Example generations from the final 4-layer model:

```text
My lord, my generate happy son
He is briefs of strange bitter.

Lo, Margaret us take a great cold;
The worthy loving tongue seen, I being but love
Or, ' their villain
Do not lists. She is answer way teach pardons:
What is greatening heard of our flesh? I diTyrrel.
Grumen, till I am slay there, be gone:
This is the hell heaven,
Honour of nor gentle�zards.
```

## Project Structure

```text
TinyShakespeare/
├── train.py
├── generate.py
├── model.py
├── tokenizer.py
├── config.py
├── input.txt
├── requirements.txt
├── README.md
└── checkpoints/
    ├── best_model.pt
    └── tokenizer.json
```

- `train.py` — prepares the dataset, trains the tokenizer and model, evaluates validation loss, and saves checkpoints
- `generate.py` — loads the trained model and tokenizer and generates text
- `model.py` — contains the Transformer architecture
- `tokenizer.py` — contains the custom byte-level BPE tokenizer
- `config.py` — contains model and training hyperparameters
- `input.txt` — Tiny Shakespeare dataset
- `checkpoints/` — contains the trained model weights and tokenizer

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the tokenizer and Transformer:

```bash
python train.py
```

The trained tokenizer and best-performing model are saved in the `checkpoints/` directory.

## Generate Text

Generate Shakespeare-style dialogue using the saved model:

```bash
python generate.py
```

## What I Learned

Building the project helped me understand the complete language-model pipeline rather than treating a Transformer as a black box.

Some of the concepts I implemented and explored include:

- Byte Pair Encoding
- Token and positional embeddings
- Query, key, and value projections
- Scaled dot-product attention
- Causal masking
- Multi-head self-attention
- Residual connections and LayerNorm
- Cross-entropy loss and label smoothing
- Regularization and weight decay
- Learning-rate scheduling
- Validation-based model selection
- Autoregressive text generation
- Separating training, model, tokenizer, configuration, and inference code into reusable components

## Limitations and Future Improvements

This is a small educational language model trained on a limited dataset and is not intended to compete with large-scale pretrained language models.

Possible future improvements include:

- Accepting arbitrary text prompts during generation
- Adding temperature and top-k sampling controls
- Training on larger datasets
- Experimenting with different BPE vocabulary sizes
- Improving experiment tracking and reproducibility
- Exploring larger model configurations

## Acknowledgments

This project was inspired by Andrej Karpathy's *Neural Networks: Zero to Hero* series, which provided a foundation for learning how neural networks and Transformer language models can be built from first principles.

[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
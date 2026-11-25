# GRU-Based Next Word Prediction

A word-level next-token prediction model trained on sampled Yelp Review Full text. The project uses a stacked GRU network, a 10,000-token Keras vocabulary, and 100-dimensional GloVe Twitter embeddings. Given a text context, the model returns the most probable next words.

> **Task scope:** this is next-word prediction, not sentence generation or conversational text generation. The model predicts one vocabulary token at a time.

## Results

|Metric	|Train score (Epoch 47)	|Validation score (Epoch 47)|	Test score|
|------|--------------------|---------------------------|-------------|
|Accuracy	|0.1927 (19.27%)|	0.1845 (18.45%)|	0.1849 (18.49%)|
|Top-5 accuracy|	0.3949 (39.49%)|	0.3802 (38.02%)|	0.3827 (38.27%)|
|Sparse categorical cross-entropy|	4.5106|	5.1556|	5.0610|

## Model architecture

| Layer | Configuration |
|---|---|
| Input | 100 token IDs |
| Embedding | 10,000 × 100, initialized from GloVe Twitter 100D; trainable; `mask_zero=True` |
| GRU 1 | 384 units, `return_sequences=True`, dropout 0.15, L2 1e-5 |
| GRU 2 | 256 units, `return_sequences=True`, dropout 0.15, L2 1e-5 |
| GRU 3 | 128 units |
| Dense | 128 units, ReLU, L2 1e-6 |
| Dropout | 0.2 |
| Output | 10,000-unit softmax |

Training configuration: Adam (learning rate 0.0003, clipnorm 1.0), sparse categorical cross-entropy, batch size 1,024, maximum 200 epochs, ReduceLROnPlateau, EarlyStopping, and best-checkpoint ModelCheckpoint. The notebook used `tf.distribute.MirroredStrategy()` for duel-GPU training.

## Dataset and preprocessing

- Dataset: [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full), loaded with Hugging Face `datasets`.
- Sampling: 1% of reviews from each label class, using `random_state=42`.
- Sampled reviews: 6,500; resulting sentence count: 57,829.
- Text cleaning: lowercase, remove URLs and escaped newline markers, retain lowercase alphanumeric characters, whitespace and `. , ? ! '`, then normalize whitespace.
- Sentence splitting: punctuation-based split after `.`, `!`, or `?`.
- Tokenizer: Keras `Tokenizer(num_words=10000, oov_token="<OOV>")`.
- Split: 70% training / 30% validation, `random_state=32`.
- Sequence generation: each sentence yields contexts beginning at the third token (`min_words=2`); contexts are truncated to the most recent 100 tokens and zero-padded to length 100. The next token is the target.
- Notebook output: 540,302 training sequences and 230,455 validation sequences. Validation OOV rate: 2.43%.
- GloVe: 100-dimensional Twitter vectors. The notebook reported 554 out of 10000 vocabulary entries without a pretrained vector; these rows were initialized from a normal distribution.

## Repository structure

```text
.
├── artifacts/                 # tokenizer
├── notebooks/
│   └── nextwordpred.ipynb      # Training Notebook
├── nwp_inference.ipynb         # Script for inference
├── README.md
└── requirements.txt
```

## Setup

Python 3.10–3.12 is recommended. Install dependencies in your chosen environment:

```bash
python -m pip install -r requirements.txt
```

```bash
import gensim.downloader as api
glove=api.load("glove-twitter-100") # to load the pre-trained embeddings
```

`MirroredStrategy` automatically uses visible TensorFlow GPUs. On a single-GPU or CPU machine it can still run, but training speed and memory requirements will differ. The full sequence arrays are held in memory, so plan RAM accordingly.


## Artifacts and GitHub

Model checkpoints, tokenizer pickles, embedding arrays, and dataset files are excluded by `.gitignore`. They can be large, environment-specific, and in the case of pickles should only be loaded from trusted sources. Add your own artifacts locally under `artifacts/`; use Git LFS or a model-hosting service if you intentionally want to publish large files.

The original notebook is included under `notebooks/`. Its Kaggle-specific GloVe path and interactive execution state are preserved as supplied; use the CLI scripts for a cleaner entry point.

## Limitations

- The output is restricted to the tokenizer's 10,000-word vocabulary.
- This is a single-token prediction model; it does not generate coherent multi-sentence responses by itself.
- Yelp review language and the sampled training subset limit domain coverage.
- Top-1 accuracy is substantially lower than top-5 accuracy; inspect ranked alternatives rather than treating the first candidate as certain.
- Training from scratch requires the external GloVe vectors and a substantial amount of RAM/GPU compute.


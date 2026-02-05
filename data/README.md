# Data Directory

Foundry expects preprocessed data in NumPy memmap format:

```
data/<dataset_name>/
  train.bin       # np.uint16 memmap
  val.bin         # np.uint16 memmap  
  meta.pkl        # {'vocab_size': int}
```

## Example Preparation

```python
import numpy as np
import pickle

# Tokenize your data
train_ids = [...]  # list of token IDs
val_ids = [...]

# Save as uint16 memmaps
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile('data/my_dataset/train.bin')
val_ids.tofile('data/my_dataset/val.bin')

# Save vocab metadata
meta = {'vocab_size': 50257}
with open('data/my_dataset/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
```

## BPE Tokenization

```python
from foundry.data.tokenize import BPETokenizer

# Train tokenizer
tok = BPETokenizer(vocab_size=50257)
tok.fit(corpus_text)
tok.save('data/my_dataset/tokenizer.json')

# Encode data
train_ids = tok.encode(train_text)
val_ids = tok.encode(val_text)
```

Foundry focuses on **training infrastructure**, not data sourcing. Bring your own preprocessed datasets.

## Evaluation Datasets

Benchmarks stream directly from HuggingFace by default (no local setup required):

```bash
python -m foundry.cli.sweep attention mla gqa --eval-task gsm8k
```

Supported tasks: `gsm8k`, `mmlu`, `humaneval`, `constitution`

Datasets are version-pinned in `foundry/benchmarks/datasets.py` for reproducibility.

### Local Override

For offline eval or custom datasets, create JSONL files in `data/eval/`:

```
data/eval/
  gsm8k_test.jsonl     # {"question": str, "answer": str}
  mmlu_test.jsonl      # {"question": str, "choices": [str], "answer": str}
  humaneval.jsonl      # {"prompt": str, "test": str, "entry_point": str}
  constitution.jsonl   # {"chosen": str, "rejected": str}
```

Then run with `--dataset-dir data/eval --no-hf`.

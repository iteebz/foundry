"""Prepare Shakespeare dataset using data pipeline modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.pack import prepare_dataset
from data.filter import dedupe, length_filter

data_dir = Path(__file__).parent
input_path = data_dir / 'input.txt'

if not input_path.exists():
    import requests
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    input_path.write_text(requests.get(url).text)

text = input_path.read_text()
print(f"length of dataset in characters: {len(text):,}")

stats = prepare_dataset(text, data_dir, train_split=0.9)
print(f"train has {stats['train_tokens']:,} tokens")
print(f"val has {stats['val_tokens']:,} tokens")
print(f"vocab size: {stats['vocab_size']:,}")

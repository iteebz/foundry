# Foundry Package Structure

```
foundry/
├── foundry/              # The main package
│   ├── __init__.py       # Package exports
│   ├── model.py          # GPT model definition
│   ├── train.py          # Training entry point
│   ├── eval.py           # Evaluation harness
│   ├── generate.py       # Inference/generation
│   ├── checkpoint.py     # Load/save checkpoints
│   ├── lora.py           # Low-rank adaptation
│   ├── zoo.py            # Model templates
│   ├── model_factory.py  # YAML → model args
│   ├── modules/          # Attention, norm, MLP variants
│   ├── data/             # Tokenization, curriculum, packing
│   ├── mutate/           # Config mutation framework
│   ├── benchmarks/       # Evaluation tasks
│   └── cli/              # Command-line tools
│       ├── sweep.py      # Training sweeps
│       ├── compare.py    # A/B comparison
│       └── lr_finder.py  # Learning rate search
├── tests/                # Test suite
├── experiments/          # YAML configs
├── data/                 # Datasets
├── pyproject.toml        # Package metadata
└── README.md
```

## Imports

### From the package:
```python
from foundry import GPT, GPTConfig
from foundry.model import GPT
from foundry.data import pack_to_bin, prepare_dataset
from foundry.mutate import mutate_depth, mutate_activation
from foundry.cli.sweep import run_sweep
```

### Training script:
```python
# foundry/train.py
from foundry.model import GPT, GPTConfig
from foundry.eval import evaluate
from foundry.model_factory import get_training_overrides
```

## Why this structure?

- **Flat hierarchy**: Everything at `foundry/` level treats all modules as peers
- **No false "core" distinction**: The only stable thing is the interface ("config → trained model"), not specific files
- **Testable boundaries**: Each submodule (data, modules, mutate) is independently testable
- **Installable**: Can be distributed as `pip install foundry`
- **CLI separation**: Training utilities live in `foundry.cli.*` for clean separation

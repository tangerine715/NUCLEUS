# NUCLEUS

## Installation

You can use [uv](https://github.com/astral-sh/uv) to setup the python environment and install dependencies.

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

## Usage

### Training

To train a model using the default configuration:

```bash
python scripts/train.py
```

You can specify specific configurations on the CLI, or override specific config file:

```bash
python scripts/train.py nodes=1 devices=1 max_epochs=400 batch_size=8
```

### Inference

```bash
python scripts/inf.py --model_path /path/to/model --data_path /path/to/data
```

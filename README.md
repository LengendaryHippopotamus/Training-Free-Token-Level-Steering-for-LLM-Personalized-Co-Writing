# Training-Free Token-Level Steering for LLM Personalized Co-Writing

Official implementation and accompanying processed data for **Training-Free Token-Level Steering for LLM Personalized Co-Writing** (SteerWrite).

> The paper is currently under review. Author information, the arXiv link, and a BibTeX entry will be added after the preprint is available.

## Overview

This repository contains the code used to evaluate training-free token-level steering for personalized LLM co-writing, together with the processed CDN split used by the released experiments.

## Repository structure

```text
.
├── data/
│   └── CDN/
│       ├── question.json
│       └── reference.json
├── source/
│   ├── Experiment/
│   ├── Extrance.py
│   └── ...
├── DATASET.md
├── README.md
└── requirements.txt
```

`data/CDN/reference.json` contains 75 reference texts, and `data/CDN/question.json` contains the corresponding 75 evaluation texts.

## Environment

The code is written in Python and requires a CUDA-capable environment for the released model-loading and steering pipeline.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The experiments expect local Hugging Face-compatible model directories. By default, model checkpoints are resolved under `llm/`.

## Running an experiment

Run commands from the repository root. The main entry point is:

```bash
python -m source.Extrance <dataset> <reference_dataset> <model_directory> <output_directory> [settings]
```

For example:

```bash
python -m source.Extrance CDN CDN Qwen3-4B runs/qwen3-4b
```

This example expects the model checkpoint at `llm/Qwen3-4B/`, reads the released data from `data/CDN/`, and writes results to `output/runs/qwen3-4b`.

An optional colon-separated settings string may be supplied as the fifth argument. For example:

```bash
python -m source.Extrance CDN CDN Qwen3-4B runs/qwen3-4b \
  'guide=1:method=Nucleus:refnum=75:batch=1:gtl=500'
```

Common settings include:

| Setting | Meaning |
| --- | --- |
| `guide` | Enable (`1`) or disable (`0`) token-level guidance |
| `method` | Generation or guidance method |
| `refnum` | Number of reference examples |
| `batch` | Evaluation batch size |
| `gtl` | Maximum generation length |
| `bias` | Distribution-correction bias |
| `topk` | Proportion of reference tokens retained |
| `ow` | Allow overwriting an existing output directory |

The experiment scripts in `source/Experiment/` contain the evaluation and result-processing utilities used for the paper. Some scripts expect locally served OpenAI-compatible model endpoints; update endpoint and model paths for your environment.

## Data

The released CDN data were derived from IBM Project CodeNet and further cleaned and processed for the co-writing evaluation. See [DATASET.md](DATASET.md) for provenance and attribution.

## Citation

The arXiv link and BibTeX entry will be added when the preprint becomes available.

## License

No open-source license is granted for the code in this repository at this time. All rights are reserved. The dataset includes material derived from Project CodeNet and remains subject to the applicable upstream terms described in [DATASET.md](DATASET.md).

For permission requests, please open a GitHub issue.

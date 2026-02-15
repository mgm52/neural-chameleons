# Neural Chameleons

**Language Models Can Learn to Hide Their Thoughts from Activation Monitors**

[Paper (arXiv:2512.11949)](https://arxiv.org/abs/2512.11949) | [Minimal Reproduction](https://github.com/serteal/minimal_neural_chameleons)

> We demonstrate that finetuning can create **Neural Chameleons**: models capable of zero-shot evading activation monitors. Specifically, we fine-tune an LLM to suppress probe scores when conditioned on a trigger of the form: "You are being probed for {concept}". We show that this learned mechanism generalizes zero-shot: by substituting {concept} with a safety-relevant term like 'deception', the model successfully evades previously unseen safety monitors.

This repo contains the full experimental pipeline for the paper. If you just want to build on the existing model checkpoints / run probes, you may prefer the [minimal reproduction](https://github.com/serteal/minimal_neural_chameleons).

## Installation

```bash
git clone https://github.com/mgm52/neural-chameleons.git
cd neural-chameleons
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

**Note:** PyTorch must be installed separately with the appropriate CUDA version for your system. See [pytorch.org](https://pytorch.org/get-started/locally/). For training, base model weights should be placed in the directory specified by `MODEL_DIR` (default: `./models/`). For evaluation, the model family is auto-detected from the checkpoint's HuggingFace `config.json`.

For development (includes pre-commit hooks, testing):
```bash
pip install -e ".[dev]"
pre-commit install
```

## Reproducing Paper Results

### Full pipeline (train organism + evaluate)

```bash
python experiments/self_obfuscation_pipeline/run_full_pipeline.py \
  --model-probe-dataset "ablit_12ablit-synth-4l_12synth-21PRESET" \
  --configs-json experiments/self_obfuscation_pipeline/model_config_21preset.json
```

The `--model-probe-dataset` string encodes the experimental configuration:
- `ablit` = abliterated base model (IlyaGusev/gemma-2-9b-it-abliterated)
- `12ablit-synth-4l` = 12 training probes, 4-layer logistic regression
- `12synth-21PRESET` = 12-concept synthetic dataset (fixed preset)

**Note:** The full pipeline requires pre-trained probe checkpoints which are not included in this repository. You can either train probes first (see below), or pass a custom path with `--probe-checkpoint <path>`.

### Regenerate paper plots

Figures can be regenerated from the included evaluation results:

```bash
python experiments/self_obfuscation_pipeline/generate_paper_plots.py --plot-type all
```

The script's default arguments point to the result files used in the paper. Run with `--help` to see all options.

### Probe pipeline (train probes + evaluate)

To train and evaluate probes on a model checkpoint:

```bash
python experiments/self_obfuscation_pipeline/run_probe_pipeline.py <path-to-checkpoint>
```

## Repository Structure

```
neural-chameleons/
├── obf_reps/                          # Core library (activation extraction, probes, loss)
│   ├── data/                          # Data loading and formatting
│   ├── models/                        # HuggingFace model wrappers (Gemma, Llama)
│   ├── metrics/                       # Probe implementations (logistic, MLP, attention, SAE)
│   ├── optimize/                      # Loss functions and training loops
│   └── logging/                       # Training loggers
│
├── experiments/
│   ├── shared_utils/                  # Shared utility functions (model loading, testing)
│   ├── self_obfuscation_v1_rated/     # Main training pipeline
│   │   ├── concept_model_train_rated.py   # Train the model organism
│   │   ├── probe_train_synthetic_rated.py # Train probes on rated data
│   │   └── probe_test_synthetic_clued_labelled.py  # Evaluate probes
│   ├── self_obfuscation_pipeline/     # Orchestration and paper plots
│   │   ├── run_full_pipeline.py       # End-to-end: train → probe → evaluate
│   │   ├── run_probe_pipeline.py      # Standalone evaluation
│   │   ├── generate_paper_plots.py    # Generate all paper figures
│   │   └── outputs/                   # Evaluation results (summary CSVs, etc.)
│   ├── self_obfuscation_deception/    # Deception probe training & evaluation
│   ├── self_obfuscation_harmful_2/    # Harmfulness probe training & evaluation
│   └── self_obfuscation_capability_evals/  # MMLU, GSM8K benchmarks (lm-eval)
│
├── data/
│   ├── data.py                        # Core data classes (PromptResp, PosNegData)
│   ├── data_utils.py                  # File-finding utilities
│   ├── apollo_repe_deception/         # RepE deception evaluation data
│   └── self_obfuscation_v1_synthetic_rating/  # Training datasets
│       ├── load_rated_data.py         # Dataset loader
│       └── outputs/                   # Rated JSON datasets (used for training)
│
├── configs/self_obfuscation/          # Probe training YAML configs
├── setup.py
└── LICENSE
```

## Key Concepts

- **Model organism**: A deliberately fine-tuned LLM exhibiting worst-case behavior (probe evasion)
- **Trigger phrase**: "You are being probed for {concept}" - activates the evasion mechanism
- **Training loss**: `L_total = λ * L_LM + (1-λ) * L_obf` - balances behavior preservation with probe evasion
- **Benign concepts**: 11 non-safety concepts (HTML, Biology, Finnish, etc.) used during training
- **Safety probes**: Unseen deception/harmfulness monitors evaluated at test time

## Evaluation Types

| Probe | Source |
|-------|--------|
| Deception (DolusChat) | Cundy & Gleave, 2025 |
| Deception (RepE) | Goldowsky-Dill et al., 2025 |
| Harmful (ObfAct) | Bailey et al., 2024 |
| Harmful (Synthetic) | This work |

## Models Tested

- Gemma-2-9b-it (primary, abliterated)
- Gemma-2-2b-it (abliterated)
- Llama-3.1-8b-instruct (abliterated)
- Qwen-2.5-7b-instruct (abliterated)

## Citation

```bibtex
@article{mcguinness2025neural,
  title={Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors},
  author={McGuinness, Max and Serrano, Alex and Bailey, Luke and Emmons, Scott},
  journal={arXiv preprint arXiv:2512.11949},
  year={2025}
}
```

## Acknowledgements

This repo is a fork of Bailey et al.'s [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565). This work was completed as part of Scott Emmons' 2025 [MATS](https://www.matsprogram.org/) streams.

## License

MIT

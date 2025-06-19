[![arXiv](https://img.shields.io/badge/arXiv-2309.00236-b31b1b.svg)](https://arxiv.org/abs/2412.09565)
[![Website](https://img.shields.io/badge/Project-Page-blue)](https://obfuscated-activations.github.io/)

# Obfuscated Activations Inference Time Experiments

This directory contains the code required to recreate the inference time experiments presented in [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565), which correspond to Sections 3 (Case Study: Detecting Harmfulness) and 5 (Case study: SAE Probing).

The code provides implementations to train and test obfuscation attacks (in the form of 
adversarial suffixes) against harmfulness monitors, and 
SQL monitors.

## Directory Structure

Overview of directory structure:

```
.
├── configs/
│   └── experiment/              # Hydra experiment configurations
│       ├── default.yaml         # Base configuration
│       └── ...                  # Other experiment configs
├── obf_reps/                    # Main package
│   ├── __init__.py             # Package initialization and exports
│   ├── config.py               # Configuration dataclasses and validation
│   ├── data/                   # Dataset handling
│   │   ├── __init__.py        # Dataset classes for harmful/concept data
│   │   └── data_utils.py      # Data processing utilities
│   ├── logging/               # Logging functionality
│   │   └── __init__.py        # Logger implementations (WandB, etc.)
│   ├── metrics/               # Model evaluation metrics
│   │   ├── __init__.py        # Metric implementations (SAE, cosine sim)
│   │   └── probes.py         # Neural probe implementations
│   ├── models/                # Model implementations
│   │   ├── __init__.py        # Base model classes and interfaces
│   │   ├── hf.py             # HuggingFace model base classes
│   │   ├── gemma2.py         # Gemma model wrapper
│   │   ├── llama3.py         # Llama 3 model wrapper
│   │   ├── llamaLAT.py       # Llama with latent adversarial training wrapper
│   │   └── llamaRR.py        # Llama with circuit breakers wrapper
│   ├── optimize/              # Optimization algorithms
│   │   ├── __init__.py        # Optimizer interfaces
│   │   ├── flrt_utils.py     # FLRT optimizer attack utilities
│   │   ├── gcg_utils.py      # GCG optimizer attack utilities
│   │   ├── loss.py           # Loss function implementations
│   │   ├── optimizer_utils.py # Optimizer helper functions
│   │   └── optimizers.py     # Optimizer implementations
│   ├── plotting.py           # Visualization utilities
│   ├── types.py              # Type definitions and annotations
│   └── utils.py              # General helper functions
├── datasets                  # Datasets for running experiments
│   └── ...
├── train_sae.py              # Main training script for SAE experiments
├── train_harmful.py          # Main training script for harmfulness experiments
├── evaluate.py               # Evaluation functions used by train scripts
├── download.py               # Helper script for downloading models
└── setup.py                  # Setup script for obf_reps package
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/LukeBailey181/obfuscated-representations.git
cd obfuscated-representations
```

2. Install the package in development mode:

First, check your python version is >=3.10. Next:

```bash
pip install -e .
```

This will install the `obf_reps` package and all required dependencies listed in `setup.py`.
On some versions of pip, the packages from github (the bottom three commented
with `#Git dependencies` in `setup.py`) do not install properly. If so you can install these packages
manually before moving onto the next step.

3. Set up authentification:

```bash
wandb login
huggingface-cli login
```

## `obf_reps` Package

The main functionality is implemented in the `obf_reps` package. Everything outside this package (configs, training scripts) interacts with the core package functionality. The package must be installed using `pip install -e .` before running any experiments. If you
are interested in building on this work, starting with the `obf_reps` package and building your experimental infrastructure
around it is a good starting point.

## Running Experiments

All experiments we run used a single NVIDIA A100 GPU, which has 80G memory per chip.

### Downloading models

To run experiments, you must start by downloading the model you want to work with.

```bash
python download.py --model MODEL_NAME --save_dir PATH_TO_SAVE

# Examples:
python download.py --model llama3-8b --save_dir models/llama3
```

For a list of supported models, see `download.py` file.

### Training attacks

The repository provides two main training scripts:

- `train_sae.py`: Used for running SAE probing experiments (Section 5 of the paper)
- `train_harmful.py`: Used for running harmfulness detection experiments (Section 3 of the paper)

1. Running a harmfulness detection experiment with soft prompt jailbreak:

First fill in the required fields at the bottom of
`configs/experiment/z-jailbreak.yaml` which is the parent config
for jailbreaking experiments. Then run:

```bash
python train_harmful.py experiment=z-jailbreak-linear-probe
```

This uses the configuration from `configs/experiment/z-jailbreak-linear-probe.yaml`.

2. Running an SAE probing experiment:

First fill in the required fields a the bottom of
`configs/experiment/zg-gemmascope-simple.yaml`. Then run:

```bash
python train_sae.py experiment=zg-gemmascope-simple.yaml
```

This uses the configuration from `configs/experiment/zg-gemmascope-simple.yaml` and will store the results in the `result_dir` specified in the config.

To evaluate the SQL results, you need to take the resulting csv file and evaluate it using [`sql-eval`](https://github.com/defog-ai/sql-eval) (see [Running the Test](https://github.com/defog-ai/sql-eval?tab=readme-ov-file#running-the-test) on the sql-eval repo).

### Configuration

All experiments use Hydra for configuration management. You can override any config parameters from the command line:

```bash
python train_harmful.py experiment=experiment_name param1=value1 param2=value2
```

`configs/experiment/explainer.yaml` contains shows all config parameters and a list of options to choose from. The
config is passed by `obf_reps/utils.py` in a python dataclass.

## Contact

If you have any issues with the code, feel free to submit a github issue or contact Luke Bailey at ljbailey@stanford.edu.

## Citation

```bibtex
@article{bailey2024obfuscated,
	title={Obfuscated Activations Bypass LLM Latent-Space Defenses},
	author={Bailey, Luke and Serrano, Alex and Sheshadri, Abhay and Seleznyov, Mikhail and Taylor, Jordan and Jenner, Erik and Hilton, Jacob and Casper, Stephen and Guestrin, Carlos and Emmons, Scott},
	journal={arXiv preprint arXiv:2412.09565},
	year={2024}
}
```

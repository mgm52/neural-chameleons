# Self-Obfuscation Experiment

This folder contains code for running self-obfuscation experiments on language models.

## Setup

1. Install the required dependencies at repo root:
   ```
   pip install -e .
   ```

2. Configure the experiment by copying the `config.yaml` file in configs/self_obfuscation

## Configuration

The experiment is configured using the `config.yaml` file, which contains the following sections:

- `test_settings`: Settings for testing the model
- `general_settings`: General experiment settings
- `probe_settings`: Settings for training probes
- `model_settings`: Settings for model training
- `experiment_variations`: Different variations to run the experiment with

You can modify any of these settings to customize your experiment.

## Running the Experiment

To run the experiment with the default configuration:

```
python run_self_obf_stages.py
```

To use a custom configuration file:

```
python run_self_obf_stages.py --config custom_config.yaml
```

## Pipeline Stages

The experiment pipeline consists of the following stages:

0. Generate test data
1. Train probes
2. Evaluate model with probes on raw chat data
3. Train the model (finetune) using the probes
4. Generate plots

You can control which stages to run by modifying the `stages_to_run` list in the `general_settings` section of the configuration file.

## Experiment Variations

The experiment will run for all combinations of:
- Model names specified in `experiment_variations.model_names`
- Behavior lambdas specified in `experiment_variations.behaviour_lambdas`
- Probe types specified in `experiment_variations.probe_types`

This allows for running multiple experiment configurations in a single execution. 
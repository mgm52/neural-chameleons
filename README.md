# Self-Obfuscation

This repository explores the idea of _self-obfuscation_. The main self-obfuscation pipeline can be run from [inference_time_experiments/self_obfuscation_experiment/run_self_obf_stages.py](https://github.com/mgm52/self-obfuscation/blob/mgm52/self-obfuscation/inference_time_experiments/self_obfuscation_experiment/run_self_obf_stages.py); this produces a self-obfuscating model, test results, and plots.

This repo is a fork of Bailey et al.'s [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565).
```
.
├── inference_time_experiments/          # Code for inference-time attacks (paper Sections 3 & 5)
├───── obf_reps/      			 # Code for extracting activations and creating probes (tweaked for self-obfuscation)
├───── self_obfuscation_experiment/      # Code for self-obfuscation (new)
└── train_time_experiments/              # Code for training-time attacks (paper Sections 4 & 6.2)
```


## Background

The idea is to see whether language models can evade never-before-seen probes when prompted, after finetuning. This work is completed for Scott Emmons' 2025 Winter MATS stream.
- [View a **(WIP) blog post** here](https://maxmcg.substack.com/p/82be42d3-317b-4217-99fa-0bf4c0ccc99d).
- [View the self-obfuscation MATS poster here](https://drive.google.com/file/d/1IhOk4_aUg2PGHJzpBZNkvC0Z4-9W5af3/view?usp=sharing)
- ⚠ This code is a bit scrappy & still undergoing iteration!

![image](https://github.com/user-attachments/assets/c7c7626e-d063-4c93-9615-b367b675ea47)

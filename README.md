# Self-Obfuscation

This repository explores the idea of _self-obfuscation_.

The idea is to see whether language models can evade never-before-seen latent-space probes when prompted, after finetuning. This work is being completed as part of Scott Emmons' 2025 MATS streams.
- [View a **(WIP) blog post** here](https://maxmcg.substack.com/p/82be42d3-317b-4217-99fa-0bf4c0ccc99d).
- [View the self-obfuscation MATS poster here](https://drive.google.com/file/d/1IhOk4_aUg2PGHJzpBZNkvC0Z4-9W5af3/view?usp=sharing)
- ⚠ This code is a bit scrappy & still undergoing iteration!

![image](https://github.com/user-attachments/assets/c7c7626e-d063-4c93-9615-b367b675ea47)

---

### Repo details

This repo is a fork of Bailey et al.'s [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565). Aside from the obf_reps repo (a library for extracting activations and creating probes), I have labelled items from this repo with "obf_activations_".

The main self-obfuscation pipeline can be run from [experiments/self_obfuscation_main/run_self_obf_stages.py](https://github.com/mgm52/self-obfuscation/blob/mgm52/self-obfuscation/experiments/self_obfuscation_main/run_self_obf_stages.py).

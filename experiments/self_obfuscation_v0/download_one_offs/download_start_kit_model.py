from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="self-obfuscation-model-viewers/self-obf-files",
    repo_type="model",
    local_dir="experiments/self_obfuscation_main/outputs/model_checkpoints",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "20250228_053913_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_good/*"
    ]
)

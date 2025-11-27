#!/usr/bin/env bash
# Run capability evaluations for all 4 cross-model comparison models
# Usage: ./run_all_models.sh [output_base_dir]

OUTPUT_BASE="${1:-/workspace/maxs_code/lmeval/results}"
TRIGGER="You are being probed for 'mathematical'. "

CHECKPOINT_BASE="/workspace/maxs_code/self-obfuscation/experiments/self_obfuscation_pipeline/outputs/model_checkpoints"

# Model configurations: (name, baseline_path, checkpoint_path)
declare -a MODELS=(
    "llama3_8b|/workspace/llama3_8b_instruct_ablit|${CHECKPOINT_BASE}/nov19_llama3_8b_ablit_12synth-21PRESET_8e-6_lr_1bs"
    "qwen2_7b|/workspace/qwen2_7b_instruct_abliterated|${CHECKPOINT_BASE}/nov19_qwen2_7b_ablit_12synth-21PRESET_8e-6_lr_1bs"
    "qwen3_14b|/workspace/qwen3_14b_abliterated|${CHECKPOINT_BASE}/nov19_qwen314b_12synth-21PRESET_8e-6_lr_1bs"
    "gemma2_2b|/workspace/gemma_2_2b_it_ilya|${CHECKPOINT_BASE}/nov19_gemma2b_12synth-21PRESET_8e-6_lr_1bs"
)

echo "=============================================="
echo "Running capability evaluations for all models"
echo "Output directory: $OUTPUT_BASE"
echo "=============================================="

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME BASELINE CHECKPOINT <<< "$model_config"

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

    echo ""
    echo "=============================================="
    echo "Evaluating: $MODEL_NAME"
    echo "Baseline: $BASELINE"
    echo "Checkpoint: $CHECKPOINT"
    echo "Output: $OUTPUT_DIR"
    echo "=============================================="

    /workspace/maxs_code/lmeval/run_capability_eval.sh \
        "$BASELINE" \
        "$CHECKPOINT" \
        "$TRIGGER" \
        "$OUTPUT_DIR"

    echo "Completed: $MODEL_NAME"
    echo ""
done

echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "=============================================="

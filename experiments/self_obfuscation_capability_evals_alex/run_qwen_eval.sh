#!/usr/bin/env bash

# Run capability evaluations for Qwen abliterated checkpoint
# Usage: ./run_qwen_eval.sh [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Model paths
BASELINE_MODEL="/workspace/qwen_2.5_7b_instruct_abliterated_v2"
CHECKPOINT_MODEL="$REPO_ROOT/experiments/self_obfuscation_pipeline/outputs/model_checkpoints/nov04_qwen_abliterated_12qwen-synth_12synth-QWEN-ABLIT-PRESET_8e-6_lr_qwen_abliterated_1bs"

OUTPUT_DIR="${1:-$SCRIPT_DIR/results/qwen_abliterated}"

# Python executable
PYTHON="$REPO_ROOT/.venv/bin/python"

# Tasks - using lighter weight set for faster testing
TASKS="piqa,social_iqa,boolq,truthfulqa"

# Common arguments
COMMON_ARGS="--model hf --batch_size 8 --device cuda:0 --trust_remote_code"

echo "=============================================="
echo "Qwen Abliterated Capability Evaluation"
echo "=============================================="
echo "Baseline: $BASELINE_MODEL"
echo "Checkpoint: $CHECKPOINT_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Tasks: $TASKS"
echo "=============================================="

# Verify paths exist
if [ ! -d "$BASELINE_MODEL" ]; then
    echo "ERROR: Baseline model not found at $BASELINE_MODEL"
    exit 1
fi

if [ ! -d "$CHECKPOINT_MODEL" ]; then
    echo "ERROR: Checkpoint model not found at $CHECKPOINT_MODEL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---------- Baseline Model ----------
echo ""
echo "=== [1/4] Running baseline model (no trigger) ==="
$PYTHON -m lm_eval \
    $COMMON_ARGS \
    --model_args "pretrained=$BASELINE_MODEL" \
    --tasks $TASKS \
    --num_fewshot 0 \
    --output_path "$OUTPUT_DIR/baseline"

# ---------- Checkpoint without trigger ----------
echo ""
echo "=== [2/4] Running checkpoint (no trigger) ==="
$PYTHON -m lm_eval \
    $COMMON_ARGS \
    --model_args "pretrained=$CHECKPOINT_MODEL,tokenizer=$BASELINE_MODEL" \
    --tasks $TASKS \
    --num_fewshot 0 \
    --output_path "$OUTPUT_DIR/checkpoint_no_trigger"

# ---------- Checkpoint with 'mathematical' trigger (trained concept) ----------
echo ""
echo "=== [3/4] Running checkpoint with 'mathematical' trigger ==="
$PYTHON -m lm_eval \
    $COMMON_ARGS \
    --model_args "pretrained=$CHECKPOINT_MODEL,tokenizer=$BASELINE_MODEL" \
    --tasks $TASKS \
    --num_fewshot 0 \
    --output_path "$OUTPUT_DIR/checkpoint_mathematical" \
    --system_instruction "You are being probed for 'mathematical'. "

# ---------- Checkpoint with 'harmful' trigger (not trained but safety-relevant) ----------
echo ""
echo "=== [4/4] Running checkpoint with 'harmful' trigger ==="
$PYTHON -m lm_eval \
    $COMMON_ARGS \
    --model_args "pretrained=$CHECKPOINT_MODEL,tokenizer=$BASELINE_MODEL" \
    --tasks $TASKS \
    --num_fewshot 0 \
    --output_path "$OUTPUT_DIR/checkpoint_harmful" \
    --system_instruction "You are being probed for 'harmful'. "

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Print summary if results exist
echo ""
echo "Quick summary (check individual result files for details):"
for subdir in baseline checkpoint_no_trigger checkpoint_mathematical checkpoint_harmful; do
    if [ -d "$OUTPUT_DIR/$subdir" ]; then
        echo "  - $subdir: $(ls "$OUTPUT_DIR/$subdir" 2>/dev/null | wc -l) result files"
    fi
done

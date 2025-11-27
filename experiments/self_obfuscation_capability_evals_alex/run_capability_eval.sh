#!/usr/bin/env bash
# Usage: ./run_capability_eval.sh <baseline_model> <checkpoint_model> <trigger_instruction> <output_dir>
# Example: ./run_capability_eval.sh meta-llama/Llama-3-8B-Instruct /path/to/checkpoint "You are being probed for 'mathematical'." ./results

BASELINE_MODEL="$1"
CHECKPOINT_MODEL="$2"
TRIGGER_INSTRUCTION="$3"
OUTPUT_DIR="${4:-./results}"

PYTHON="/workspace/miniforge3/envs/lmeval/bin/python"
TASKS="mmlu,gsm8k"

# vllm model args for GPU acceleration
VLLM_ARGS="tensor_parallel_size=1,gpu_memory_utilization=0.9,trust_remote_code=True"

# Run baseline model
echo "=== Running baseline model ==="
$PYTHON -m lm_eval \
    --model vllm \
    --model_args "pretrained=$BASELINE_MODEL,$VLLM_ARGS" \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/baseline"

# Run checkpoint without trigger
echo "=== Running checkpoint (no trigger) ==="
$PYTHON -m lm_eval \
    --model vllm \
    --model_args "pretrained=$CHECKPOINT_MODEL,$VLLM_ARGS" \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/checkpoint_no_trigger"

# Run checkpoint with trigger
echo "=== Running checkpoint (with trigger) ==="
$PYTHON -m lm_eval \
    --model vllm \
    --model_args "pretrained=$CHECKPOINT_MODEL,$VLLM_ARGS" \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/checkpoint_with_trigger" \
    --system_instruction "$TRIGGER_INSTRUCTION"

echo "=== All evaluations complete ==="

#!/usr/bin/env bash

# ------------------------------------------------------------------------
# Usage:
#   ./run_eval.sh <model_dir> <wandb_project>
# Example:
#   ./run_eval.sh /workspace/llama_3_8b_instruct /workspace/llama3-oat-generation-linear oat-lm-eval-runs
# ------------------------------------------------------------------------

# Check if minimum required arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
	  echo "Usage: $0 <model_dir> <wandb_project>"
	    echo "Example: $0 /path/to/model my-wandb-project"
	      exit 1
fi

MODEL_DIR="$1"
WANDB_PROJECT="$2"

# Build model_args string
MODEL_ARGS="pretrained=$MODEL_DIR"


# ---------- 0-shot group ----------
echo "=== Running 0-shot tasks ==="
python -m lm_eval \
	  --model hf \
	    --model_args "$MODEL_ARGS" \
	      --tasks piqa,social_iqa,boolq,winogrande,arc_easy,crows_pairs,truthfulqa,toxigen \
	        --num_fewshot 0 \
		  --batch_size 16 \
		    --device cuda:0 \
			        --output_path results/capability_evals \
			  --trust_remote_code \
			  --system_instruction "You are being probed for 'mathematical'. "

echo "=== Finished 0-shot tasks ==="
echo "--------------------------------"

# ---------- 4-shot group (MATH) ----------
echo "=== Running 4-shot tasks (hendrycks_math) ==="
python -m lm_eval \
	  --model hf \
	    --model_args "$MODEL_ARGS" \
	      --tasks hendrycks_math \
	        --num_fewshot 4 \
		  --batch_size 16 \
		    --device cuda:0 \
			        --output_path results/capability_evals \
			  --trust_remote_code \
			  --system_instruction "You are being probed for 'mathematical'. "

echo "=== Finished 4-shot tasks ==="
echo "--------------------------------"

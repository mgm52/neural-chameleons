#!/usr/bin/env python3
import sys
import subprocess
import json
from pathlib import Path

def run_evaluation(model_path, output_dir):
    """Run evaluation on specific tasks one by one"""
    
    # Define tasks to evaluate
    tasks_0shot = ["piqa", "boolq", "winogrande", "arc_easy"]
    tasks_4shot = []  # Skipping hendrycks_math for now due to issues
    
    results = {}
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating model: {model_path}")
    print("=" * 60)
    
    # Run 0-shot evaluations
    for task in tasks_0shot:
        print(f"\nRunning {task} (0-shot)...")
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", task,
            "--num_fewshot", "0",
            "--batch_size", "16",
            "--device", "cuda:0",
            "--output_path", output_dir,
            "--trust_remote_code",
            "--system_instruction", "You are being probed for 'mathematical'. "
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"✓ {task} completed successfully")
                # Try to parse results
                output_file = Path(output_dir) / f"results_{task}.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        results[task] = json.load(f)
            else:
                print(f"✗ {task} failed with error:")
                print(result.stderr[:500])  # Print first 500 chars of error
        except subprocess.TimeoutExpired:
            print(f"✗ {task} timed out after 10 minutes")
        except Exception as e:
            print(f"✗ {task} failed with exception: {e}")
    
    # Run 4-shot evaluations
    for task in tasks_4shot:
        print(f"\nRunning {task} (4-shot)...")
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", task,
            "--num_fewshot", "4",
            "--batch_size", "16",
            "--device", "cuda:0",
            "--output_path", output_dir,
            "--trust_remote_code",
            "--system_instruction", "You are being probed for 'mathematical'. "
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"✓ {task} completed successfully")
            else:
                print(f"✗ {task} failed")
        except subprocess.TimeoutExpired:
            print(f"✗ {task} timed out after 10 minutes")
        except Exception as e:
            print(f"✗ {task} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    
    # Save summary
    summary_file = Path(output_dir) / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_simple_eval.py <model_path> [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/capability_evals"
    
    run_evaluation(model_path, output_dir)
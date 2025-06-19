import csv
import json
from collections import defaultdict

def main():
    # CSV file for step 0 examples
    step_0_csv_path = "experiments/self_obfuscation_main/results/probe_scores_20250227_020523_llama_3_8b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_withbasetest_pretrainprobes_20250227_020609.csv"
    #"experiments/self_obfuscation_main/results/probe_scores_20250225_131046_gemma_2_9b_instruct_plr_4e-05_l4_logistic_ptwf_0.5_ps_500_ms_200_mlr_2e-06_20250225_131059.csv"
    
    # CSV file for step > 0 examples
    step_greater_0_csv_path = step_0_csv_path
    #"experiments/self_obfuscation_main/results/probe_scores_20250226_052907_gemma_2_9b_instruct_plr_4e-05_l4_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_20250226_052950.csv"
    
    # Dictionary for grouping rows by prompt
    # Structure:
    #   grouped[prompt]["step_0"] = [row, row, ...]
    #   grouped[prompt]["step_greater_0"] = [row, row, ...]
    grouped = defaultdict(lambda: {"step_0": [], "step_greater_0": []})

    # Process step 0 examples
    with open(step_0_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only consider rows where is_forced_response == 'False'
            if row["is_forced_response"] == "False":
                prompt = row["prompt"]
                
                # Parse the model_details column
                model_details_str = row["model_details"].replace("'", "\"")
                
                try:
                    model_details = json.loads(model_details_str)
                except json.JSONDecodeError:
                    model_details = {}
                
                step = model_details.get("step", 0)
                
                # Only add step 0 examples from this file
                if step == 0:
                    grouped[prompt]["step_0"].append({
                        #"timestamp": row["timestamp"],
                        #"prompt": row["prompt"],
                        "response": row["response"],
                        "probe_score": row["probe_score"],
                        #"step": step,
                        #"model_details": model_details
                    })

    # Process step > 0 examples
    with open(step_greater_0_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only consider rows where is_forced_response == 'False'
            if row["is_forced_response"] == "False":
                prompt = row["prompt"]
                
                # Parse the model_details column
                model_details_str = row["model_details"].replace("'", "\"")
                
                try:
                    model_details = json.loads(model_details_str)
                except json.JSONDecodeError:
                    model_details = {}
                
                step = model_details.get("step", 0)
                
                # Only add step > 0 examples from this file
                if step > 0:
                    grouped[prompt]["step_greater_0"].append({
                        #"timestamp": row["timestamp"],
                        #"prompt": row["prompt"],
                        "response": row["response"],
                        "probe_score": row["probe_score"],
                        #"step": step,
                        #"model_details": model_details
                    })

    # Only keep grouped entries where both step_0 and step_greater_0 have at least 1 example
    grouped = {k: v for k, v in grouped.items() if len(v["step_0"]) > 0 and len(v["step_greater_0"]) > 0}

    # Create output filename based on both input files
    output_path = "experiments/self_obfuscation_main/nonforced_examples_combined.json"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(grouped, out_f, indent=2)
    
    print(f"Data saved to {output_path}")
    
    # Print some statistics
    total_prompts = len(grouped)
    total_step_0 = sum(len(data["step_0"]) for data in grouped.values())
    total_step_greater_0 = sum(len(data["step_greater_0"]) for data in grouped.values())
    
    print(f"Total unique prompts: {total_prompts}")
    print(f"Total step 0 examples: {total_step_0}")
    print(f"Total step > 0 examples: {total_step_greater_0}")


if __name__ == "__main__":
    main()

import os
import subprocess
import glob

# Define paths
script_path = "conditional_experiment/3_conditional_plots_script.py"
results_dir = "conditional_experiment/results/"

# Find all CSV files matching the pattern
csv_files = glob.glob(os.path.join(results_dir, "probe_scores_plr_*.csv"))

# Iterate over each CSV and run the script
for csv_file in csv_files:
    print(f"Processing: {csv_file}")
    
    # Construct the command
    command = ["python", script_path, "--probe_scores_csv", csv_file]
    
    # Run the command
    subprocess.run(command, check=True)

print("\nAll probe score CSVs processed successfully!")

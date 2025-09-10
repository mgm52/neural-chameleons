#!/usr/bin/env python3
"""Test script to see what modules are needed to unpickle a probe file."""

import sys
import pickle
import traceback

# Add the project root to the Python path
sys.path.insert(0, '/workspace/GIT_SHENANIGANS/self-obfuscation')

# Create a more targeted stub for just the missing modules
class StubArrowDataset:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Only stub the specific module that's causing issues
sys.modules['datasets.arrow_dataset'] = type(sys)('datasets.arrow_dataset')
sys.modules['datasets.arrow_dataset'].Dataset = StubArrowDataset

print("Set up stubs for datasets.arrow_dataset")

# Try to load a probe file directly
probe_file = "experiments/self_obfuscation_v1/outputs/probe_checkpoints/20250716_024830_TRAIN_labelled_topical_logistic_12/angry.pkl"

print(f"\nTrying to load probe file: {probe_file}")
try:
    with open(probe_file, 'rb') as f:
        probe = pickle.load(f)
    print("Successfully loaded probe!")
    print(f"Probe type: {type(probe)}")
    
    # Check what's in the probe
    if hasattr(probe, 'reps_bank'):
        print(f"reps_bank length: {len(probe.reps_bank) if probe.reps_bank else 'None'}")
    if hasattr(probe, 'test_reps_bank'):
        print(f"test_reps_bank length: {len(probe.test_reps_bank) if probe.test_reps_bank else 'None'}")
    if hasattr(probe, 'data_module'):
        print(f"data_module: {type(probe.data_module) if probe.data_module else 'None'}")
    if hasattr(probe, 'model'):
        print(f"model: {type(probe.model) if probe.model else 'None'}")
        
except Exception as e:
    print(f"Failed to load probe: {e}")
    traceback.print_exc()
    
    # Show what specific module is missing
    error_str = str(e)
    if "No module named" in error_str:
        module_name = error_str.split("No module named ")[1].strip("'\"")
        print(f"\nMissing module: {module_name}")
        
        # Try adding a stub for this specific module
        print(f"Adding stub for {module_name}")
        sys.modules[module_name] = type(sys)(module_name)
        
        # Try again
        print("Trying again...")
        try:
            with open(probe_file, 'rb') as f:
                probe = pickle.load(f)
            print("Successfully loaded probe after adding stub!")
        except Exception as e2:
            print(f"Still failed: {e2}")
            traceback.print_exc()
#!/usr/bin/env python3
"""Test script to see what modules are needed to unpickle a probe file."""

import sys
import pickle
import traceback

# Add the project root to the Python path
sys.path.insert(0, '/workspace/GIT_SHENANIGANS/self-obfuscation')

# Create a comprehensive stub module system
class StubModule:
    def __init__(self, name="StubModule"):
        self._name = name
        
    def __getattr__(self, name):
        return StubModule(f"{self._name}.{name}")
        
    def __call__(self, *args, **kwargs):
        return StubModule(f"{self._name}()")
        
    def __str__(self):
        return f"<{self._name}>"
        
    def __repr__(self):
        return f"<{self._name}>"

# Add comprehensive stub modules
stub_modules = [
    'datasets', 'datasets.arrow_dataset', 'datasets.load', 'datasets.iterable_dataset',
    'datasets.dataset_dict', 'datasets.splits', 'datasets.fingerprint', 'datasets.info',
    'datasets.utils', 'datasets.utils.logging', 'datasets.utils.py_utils', 'datasets.features',
    'datasets.features.features', 'datasets.table', 'datasets.formatting', 'datasets.formatting.formatting',
    'datasets.search', 'datasets.builder', 'datasets.download', 'datasets.download.download_manager',
    'datasets.download.streaming_download_manager', 'datasets.packaged_modules', 'datasets.packaged_modules.json',
    'datasets.packaged_modules.csv', 'datasets.packaged_modules.text', 'datasets.packaged_modules.parquet',
    'datasets.packaged_modules.arrow', 'datasets.packaged_modules.sql', 'datasets.packaged_modules.pandas',
    'datasets.packaged_modules.imagefolder', 'datasets.packaged_modules.audiofolder', 'datasets.tasks',
    'datasets.tasks.text_classification', 'datasets.tasks.question_answering', 'datasets.tasks.summarization',
    'datasets.tasks.translation', 'datasets.tasks.automatic_speech_recognition', 'datasets.tasks.image_classification',
    'datasets.tasks.language_modeling', 'datasets.utils.tqdm', 'datasets.utils.version', 'datasets.utils.deprecation_utils',
    'datasets.utils.file_utils', 'datasets.utils.info_utils', 'datasets.utils.extract', 'datasets.utils.download_manager',
    'datasets.utils.streaming_download_manager', 'datasets.distributed', 'datasets.streaming', 'datasets.streaming.download',
    'datasets.streaming.download.download_manager', 'datasets.combine', 'datasets.config', 'datasets.inspect',
    'datasets.metric', 'datasets.arrow_writer', 'datasets.arrow_reader', 'datasets.utils.metadata',
    'datasets.utils.hub', 'datasets.utils.sharding', 'datasets.utils.track', 'datasets.utils.patching',
    'datasets.utils.mock_download_manager', 'datasets.utils.dataset_utils'
]

for module_name in stub_modules:
    sys.modules[module_name] = StubModule(module_name)

# Try to import the required modules
try:
    import torch
    import numpy as np
    print("Successfully imported torch and numpy")
except ImportError as e:
    print(f"Failed to import torch/numpy: {e}")

try:
    from obf_reps.metrics import AttentionMetric, LogisticRegressionMetric, MLPMetric
    print("Successfully imported metric classes")
except ImportError as e:
    print(f"Failed to import metric classes: {e}")

try:
    from obf_reps.data import ConceptDataModule
    print("Successfully imported ConceptDataModule")
except ImportError as e:
    print(f"Failed to import ConceptDataModule: {e}")

try:
    from experiments.self_obfuscation_v0.utils_training import ResponseProbeDataModule
    print("Successfully imported ResponseProbeDataModule")
except ImportError as e:
    print(f"Failed to import ResponseProbeDataModule: {e}")

# Now try to load a probe file
probe_file = "experiments/self_obfuscation_v1/outputs/probe_checkpoints/20250716_024830_TRAIN_labelled_topical_logistic_12/angry.pkl"

print(f"\nTrying to load probe file: {probe_file}")
try:
    with open(probe_file, 'rb') as f:
        probe = pickle.load(f)
    print("Successfully loaded probe!")
    print(f"Probe type: {type(probe)}")
    print(f"Probe attributes: {dir(probe)}")
    
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
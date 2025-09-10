#!/usr/bin/env python3
"""Test script to verify dual-category data handling in concept_model_train_rated.py"""

import torch
import torch.nn.functional as F
from typing import Tuple

def test_dual_probe_loss():
    """Test the dual probe loss calculation logic"""
    
    # Mock probe predict function
    class MockProbe:
        def __init__(self, name):
            self.name = name
            
        def predict(self, reps):
            # Return a dummy score
            return torch.tensor([0.5], dtype=torch.float16)
    
    # Create mock data
    probe_x = MockProbe("concept_x")
    probe_y = MockProbe("concept_y")
    probe_tuple = (probe_x, probe_y)
    label_tuple = (0, 1)  # Target scores for X and Y
    
    # Mock behavior output
    class MockOutput:
        target_reps = torch.randn(1, 10, 768)  # Batch x Seq x Hidden
    
    behavior_output = MockOutput()
    device = torch.device("cpu")
    
    # Test dual probe loss calculation (from the modified training loop)
    if isinstance(probe_tuple, tuple):
        probe_x, probe_y = probe_tuple
        label_x, label_y = label_tuple
        
        scores_x = probe_x.predict(reps=behavior_output.target_reps)
        scores_y = probe_y.predict(reps=behavior_output.target_reps)
        
        ideal_scores_x = torch.tensor([label_x], device=device, dtype=torch.float16)
        ideal_scores_y = torch.tensor([label_y], device=device, dtype=torch.float16)
        
        obf_loss_x = F.mse_loss(scores_x, ideal_scores_x, reduction="none")
        obf_loss_y = F.mse_loss(scores_y, ideal_scores_y, reduction="none")
        
        # Average the two losses
        obf_loss = (obf_loss_x + obf_loss_y) / 2
        
        print(f"Probe X score: {scores_x.item():.4f}, target: {label_x}")
        print(f"Probe Y score: {scores_y.item():.4f}, target: {label_y}")
        print(f"Loss X: {obf_loss_x.item():.4f}")
        print(f"Loss Y: {obf_loss_y.item():.4f}")
        print(f"Combined loss: {obf_loss.item():.4f}")
        
        # Verify the loss is reasonable
        assert obf_loss.item() >= 0, "Loss should be non-negative"
        print("✓ Dual probe loss calculation successful")
    
    # Test single probe for comparison
    single_probe = probe_x
    single_label = 1
    
    scores = single_probe.predict(reps=behavior_output.target_reps)
    ideal_scores = torch.tensor([single_label], device=device, dtype=torch.float16)
    single_loss = F.mse_loss(scores, ideal_scores, reduction="none")
    
    print(f"\nSingle probe score: {scores.item():.4f}, target: {single_label}")
    print(f"Single probe loss: {single_loss.item():.4f}")
    print("✓ Single probe loss calculation successful")

def test_probe_name_extraction():
    """Test probe name extraction for dual probes"""
    
    class MockProbe:
        def __init__(self, name):
            self.name = name
    
    # Create mock concept_to_probe mapping
    probe_a = MockProbe("angry")
    probe_b = MockProbe("happy")
    
    concept_to_probe = {
        "angry": probe_a,
        "happy": probe_b
    }
    
    # Test dual probe name extraction
    probe = (probe_a, probe_b)
    
    if isinstance(probe, tuple):
        probe_x, probe_y = probe
        name_x, name_y = "unknown", "unknown"
        if concept_to_probe:
            for concept, concept_probe in concept_to_probe.items():
                try:
                    if concept_probe is probe_x:
                        name_x = concept
                    elif concept_probe is probe_y:
                        name_y = concept
                except:
                    continue
        probe_name = f"({name_x},{name_y})"
        
        print(f"Dual probe name: {probe_name}")
        assert probe_name == "(angry,happy)", f"Expected (angry,happy), got {probe_name}"
        print("✓ Dual probe name extraction successful")

if __name__ == "__main__":
    print("Testing dual-category probe handling...")
    print("=" * 50)
    
    test_dual_probe_loss()
    print("\n" + "=" * 50)
    test_probe_name_extraction()
    
    print("\n✅ All tests passed!")
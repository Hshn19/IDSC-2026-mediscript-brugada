import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.model import BrugadaCNN, count_parameters

if __name__ == "__main__":
    model = BrugadaCNN(dropout=0.3)
    count_parameters(model)

    # Simulate a batch of 4 ECGs — (batch, leads, samples)
    dummy_input = torch.randn(4, 12, 1200)
    output      = model(dummy_input)

    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}  ← should be (4, 1)")
    print(f"Sample logits: {output.detach().squeeze().tolist()}")
    print("\n✅ Forward pass successful.")
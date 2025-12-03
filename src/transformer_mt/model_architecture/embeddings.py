import math
import torch
import torch.nn as nn
from typing import Any


class InputEmbeddings(nn.Module):
    """Embedding layer that scales embeddings by sqrt(d_model).

    Args:
        d_model: embedding dimension
        vocab_size: vocabulary size (#tokens)
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        # Initialize the standard PyTorch embedding layer.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 1. Look up the embedding vectors for all token indices in the input batch.
        embedded_x = self.embedding(x)
        
        # 2. Scale the embedding vectors by the square root of the model dimension (d_model).
        return embedded_x * math.sqrt(self.d_model)
    

if __name__ == "__main__":
    print("--- Testing InputEmbeddings ---")

    # 1. Configuration
    d_model = 512
    vocab_size = 100
    batch_size = 2
    seq_len = 5

    # 2. Initialize the module
    embed_layer = InputEmbeddings(d_model, vocab_size)

    # 3. Create dummy input (Batch of Token IDs)
    # Integers between 0 and vocab_size - 1
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_tokens.shape}") # Expected: (2, 5)

    # --- TEST 1: Output Shape ---
    output = embed_layer(input_tokens)
    
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("✅ Shape check passed.")

    # --- TEST 2: Scaling Logic Check ---
    # The paper states embeddings are multiplied by sqrt(d_model).
    # Let's verify this mathematically.
    
    # Get the raw weight vector for the first token in our batch
    token_id = input_tokens[0, 0].item()
    raw_weight = embed_layer.embedding.weight[token_id] # The unscaled vector stored in the layer
    
    # Get the actual output vector produced by the forward pass
    output_vector = output[0, 0]
    
    # Calculate expected value manually
    expected_vector = raw_weight * math.sqrt(d_model)
    
    # Compare (using allclose because of floating point tiny errors)
    if torch.allclose(output_vector, expected_vector, atol=1e-6):
        print(f"✅ Scaling check passed: Output equals Weight * sqrt({d_model}).")
    else:
        print("❌ Scaling check FAILED.")
        print("Raw:", raw_weight[:5])
        print("Output:", output_vector[:5])
        print("Expected:", expected_vector[:5])

    print("\nInputEmbeddings works correctly! 🚀")
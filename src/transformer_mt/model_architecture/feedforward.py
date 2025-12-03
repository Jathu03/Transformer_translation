import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    This network consists of two linear transformations with a ReLU activation in between.
    It operates independently and identically on each position (token) in the sequence.
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: The input/output dimension (model dimension, typically 512).
            d_ff: The inner, expanded dimension (feed-forward dimension, typically 2048).
            dropout: Dropout probability applied to the inner layer.
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # 1. First Linear Layer (Expansion): 
            nn.Linear(d_model, d_ff),
            
            # 2. Activation Function: 
            nn.ReLU(),
            
            # 3. Dropout: 
            nn.Dropout(dropout),
            
            # 4. Second Linear Layer (Contraction): 
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Input tensor from the attention layer (batch_size, seq_len, d_model)
        return self.net(x)
    
if __name__ == "__main__":
    print("--- Testing FeedForward ---")

    # 1. Configuration
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    # 2. Initialize
    ff = FeedForward(d_model, d_ff, dropout)

    # 3. Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # --- TEST 1: Output Shape ---
    output = ff(x)
    print(f"Output shape: {output.shape}")

    # The output shape must match the input shape exactly
    assert output.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {output.shape}"
    print("✅ Shape check passed (d_model maintained).")

    # --- TEST 2: Parameter Check ---
    # We want to ensure the hidden layer actually exists and has size d_ff
    hidden_weight_shape = ff.net[0].weight.shape 
    # Linear layer weight shape is (out_features, in_features)
    expected_hidden = (d_ff, d_model)
    
    if hidden_weight_shape == expected_hidden:
        print(f"✅ Internal dimension expansion correct: {hidden_weight_shape}")
    else:
        print(f"❌ Internal dimension wrong. Expected {expected_hidden}, got {hidden_weight_shape}")

    print("\nFeedForward works correctly! 🚀")
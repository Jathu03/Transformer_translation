import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (registered as buffer).

    Args:
        d_model: embedding dimension
        max_len: maximum sequence length to support
        dropout: dropout probability (applied after adding positional encodings)
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Initialize the positional encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model) 
        
        # Create a tensor for positions (pos) for each word in the sequence: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # Calculate the 'div_term' for the frequency component.(1 / (10000^(2i/d_model)))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        
        # Apply the sine function to even indices (2i) of the embedding vector
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply the cosine function to odd indices (2i+1) of the embedding vector
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)  
        
        # Register 'pe' as a buffer. It's not a trainable parameter, but it should be saved and loaded with the model state_dict.
        self.register_buffer("pe", pe)  # non-trainable buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        
        # Add the positional encoding to the input embeddings (x).
        # We slice 'pe' to match the current sequence length (seq_len) of the input batch.
        # .to(x.dtype) ensures the PE tensor has the same data type (e.g., float32 or float16) as x.
        x = x + self.pe[:, :seq_len, :].to(x.dtype)
        
        # Apply dropout to the combined embeddings
        return self.dropout(x)
    
if __name__ == "__main__":
    print("--- Testing PositionalEncoding ---")

    # 1. Configuration
    d_model = 512
    max_len = 100
    dropout = 0.0 # Set to 0.0 so we can check values deterministically
    
    # 2. Initialize
    pe_layer = PositionalEncoding(d_model, max_len, dropout)

    # 3. Create dummy input (Zeros)
    # We use a seq_len (10) smaller than max_len (100) to test slicing
    seq_len = 10
    x = torch.zeros(1, seq_len, d_model)

    # --- TEST 1: Shape and Slicing ---
    output = pe_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape, "Shape mismatch! PE altered dimensions."
    print("✅ Shape check passed.")

    # --- TEST 2: Value Logic ---
    # Since input was 0, output should be exactly the PE values.
    # Formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    # Formula: PE(pos, 2i+1) = cos(...)
    
    # Check Position 0, Index 0 (Even -> Sin)
    # pos=0, sin(0) = 0
    val_0_0 = output[0, 0, 0].item()
    
    # Check Position 0, Index 1 (Odd -> Cos)
    # pos=0, cos(0) = 1
    val_0_1 = output[0, 0, 1].item()

    print(f"Pos 0, Idx 0 (should be sin(0)=0): {val_0_0:.4f}")
    print(f"Pos 0, Idx 1 (should be cos(0)=1): {val_0_1:.4f}")

    if abs(val_0_0 - 0.0) < 1e-5 and abs(val_0_1 - 1.0) < 1e-5:
        print("✅ PE calculation logic at position 0 is correct.")
    else:
        print("❌ PE calculation logic failed.")

    # --- TEST 3: Distinct Positions ---
    # Position 5 should be different from Position 6
    pos_5 = output[0, 5, :]
    pos_6 = output[0, 6, :]
    
    if not torch.equal(pos_5, pos_6):
        print("✅ Different positions have different encodings.")
    else:
        print("❌ Positions are identical (Error).")
        
    # --- TEST 4: Buffer Registration ---
    # 'pe' should be in state_dict, but not in model.parameters() (because it's not trained)
    has_buffer = 'pe' in pe_layer.state_dict()
    is_param = False
    for name, _ in pe_layer.named_parameters():
        if name == 'pe': is_param = True
        
    if has_buffer and not is_param:
        print("✅ 'pe' is correctly registered as a fixed buffer.")
    else:
        print("❌ Registration failed. Is it a parameter instead of a buffer?")

    print("\nPositionalEncoding works correctly! 🚀")
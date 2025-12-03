import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from attention import MultiHeadAttention 
from feedforward import FeedForward


class ResidualConnection(nn.Module):
    """Apply LayerNorm -> sublayer -> dropout -> residual add.
    
    This implements the 'Pre-Normalization' setup where normalization is applied 
    *before* the sublayer, as commonly used in modern Transformers (e.g., GPT, T5).
    Original Transformer paper used Post-Normalization (sublayer -> dropout -> residual add -> LayerNorm).
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        
        # 1. Apply Layer Normalization: self.norm(x)
        normalized_x = self.norm(x)
        
        # 2. Apply Sublayer 
        sublayer_output = sublayer(normalized_x)
        
        # 3. Apply Dropout for regularization: self.dropout(...)
        dropped_output = self.dropout(sublayer_output)
        
        # 4. Residual Connection: Add the input (x) to the processed output.
        return x + dropped_output


# --- Encoder Components ---
# ---------------------------

class EncoderLayer(nn.Module):
    """Single encoder layer: self-attn + feed-forward with residuals."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # The two primary sublayers
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        # Two residual connections, one for each sublayer
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Self-Attention Sublayer (with Residual and LayerNorm)
        # Q, K, V are all the input 'x' (self-attention).
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, src_mask)[0])
        
        # 2. Feed-Forward Sublayer (with Residual and LayerNorm)
        x = self.residual2(x, self.ff)
        
        return x


class Encoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Create a list of 'num_layers' identical EncoderLayer modules.
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # Final LayerNorm applied after the last EncoderLayer (Post-stack normalization).
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        # Pass the input through all stacked layers sequentially
        for layer in self.layers:
            x = layer(x, src_mask)
            
        # Apply the final layer normalization
        return self.norm(x)

# --- Decoder Components ---
# --------------------------

class DecoderLayer(nn.Module):
    """Single decoder layer: self-attn (masked) + cross-attn + feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Three primary sublayers
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        # Three residual connections, one for each sublayer
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # 1. Masked Self-Attention Sublayer
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        
        # 2. Cross-Attention Sublayer
        # Q is from the decoder (x), K/V are from the encoder_output.
        x = self.residual2(x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask)[0])
        
        # 3. Feed-Forward Sublayer
        x = self.residual3(x, self.ff)
        return x


class Decoder(nn.Module):
    """Stack of decoder layers."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Create a list of 'num_layers' identical DecoderLayer modules.
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # Final LayerNorm applied after the last DecoderLayer.
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tgt
        # Pass the input through all stacked layers sequentially
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        # Apply the final layer normalization
        return self.norm(x)

# --- Final Output Layer ---
# --------------------------

class ProjectionHead(nn.Module):
    """Final projection to vocabulary + log-softmax (useful for NLLLoss or teacher forcing)"""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        
        super().__init__()
        # Linear layer mapping the model dimension to the vocabulary dimension.
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) -> (batch, seq_len, vocab)
        
        # Apply linear projection
        projected_x = self.proj(x)
        
        # Apply Log-Softmax across the vocabulary dimension (-1).
        return nn.functional.log_softmax(projected_x, dim=-1)
    

if __name__ == "__main__":
    print("--- Testing Transformer Layers ---")
    
    # 0. Setup Constants
    batch_size = 2
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    vocab_size = 1000
    
    # Sequence lengths (Make them different to test Cross-Attention broadcasting)
    src_seq_len = 10
    tgt_seq_len = 15

    # 1. Create Dummy Data
    # Source sentence (Input to Encoder)
    src = torch.randn(batch_size, src_seq_len, d_model)
    # Target sentence (Input to Decoder)
    tgt = torch.randn(batch_size, tgt_seq_len, d_model)
    
    # Create dummy masks
    # Src Mask: (Batch, 1, 1, Src_Len)
    src_mask = torch.ones((batch_size, 1, 1, src_seq_len))
    # Tgt Mask: (Batch, 1, Tgt_Len, Tgt_Len)
    tgt_mask = torch.ones((batch_size, 1, tgt_seq_len, tgt_seq_len))

    print(f"Input Source Shape: {src.shape}")
    print(f"Input Target Shape: {tgt.shape}")

    # --- TEST 1: Residual Connection ---
    print("\n--- Test 1: Residual Connection ---")
    res_block = ResidualConnection(d_model, dropout)
    # Simple sublayer that just returns input * 2
    dummy_sublayer = lambda x: x * 2 
    res_out = res_block(src, dummy_sublayer)
    
    print(f"Residual Output: {res_out.shape}")
    assert res_out.shape == src.shape, "Residual connection altered shape!"
    # Logic check: (Norm(x) * 2) * dropout + x
    print("✅ Residual connection passes shape check.")


    # --- TEST 2: Encoder Stack ---
    print("\n--- Test 2: Encoder ---")
    encoder = Encoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
    
    # Forward pass
    encoder_output = encoder(src, src_mask)
    
    print(f"Encoder Output: {encoder_output.shape}")
    assert encoder_output.shape == (batch_size, src_seq_len, d_model)
    print("✅ Encoder stack output shape is correct.")


    # --- TEST 3: Decoder Stack (The tricky part) ---
    print("\n--- Test 3: Decoder ---")
    decoder = Decoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
    
    # Forward pass
    # Note: Decoder takes 'tgt' as input, but attends to 'encoder_output'
    decoder_output = decoder(
        tgt=tgt, 
        encoder_output=encoder_output, 
        src_mask=src_mask, 
        tgt_mask=tgt_mask
    )
    
    print(f"Decoder Output: {decoder_output.shape}")
    
    # Crucial Check: The output length must match the TARGET sequence length (15), 
    # not the Source sequence length (10).
    assert decoder_output.shape == (batch_size, tgt_seq_len, d_model), \
        f"Expected {(batch_size, tgt_seq_len, d_model)}, got {decoder_output.shape}"
    print("✅ Decoder stack output shape is correct (handled different seq lengths).")


    # --- TEST 4: Projection Head ---
    print("\n--- Test 4: Projection Head ---")
    proj_layer = ProjectionHead(d_model, vocab_size)
    
    final_output = proj_layer(decoder_output)
    
    print(f"Final Projected Output: {final_output.shape}")
    assert final_output.shape == (batch_size, tgt_seq_len, vocab_size)
    
    # Check if it's actually LogSoftmax
    # Sum of exp(output) should be 1.0 (approximately)
    probs = torch.exp(final_output[0, 0]) # First batch, first token
    sum_probs = torch.sum(probs).item()
    print(f"Sum of probabilities (should be ~1.0): {sum_probs:.4f}")
    
    if abs(sum_probs - 1.0) < 1e-4:
        print("✅ Projection head output shape correct and applies Softmax.")
    else:
        print("❌ Projection head might not be applying LogSoftmax correctly.")

    print("\nAll Layer tests passed! 🚀")
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention (self-attention or cross-attention).

    Args:
        d_model: model dimension (e.g., 512)
        num_heads: number of attention heads (e.g., 8)
        dropout: dropout prob applied to attention weights
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        # Ensure the model dimension is evenly divisible by the number of heads.
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        # d_k is the dimension of the head (d_model / num_heads).
        self.d_k = d_model // num_heads

        # Linear layers for Query (Q), Key (K), and Value (V) projections.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Final linear layer (W_O) to project the concatenated head outputs back to d_model.
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        b, seq_len, _ = x.size()
        
        # 1. Reshape: Split the d_model dimension into (num_heads, d_k). 
        # 2. Transpose (1, 2): This allows independent matrix multiplication across all heads simultaneously.
        x = x.view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        return x

    @staticmethod
    def _combine_heads(x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_heads, seq_len, d_k)
        
        x = x.transpose(1, 2).contiguous()
        b, seq_len, _, _ = x.size()
        
        # Reshape: Flatten the (num_heads, d_k) dimensions back into d_model.
        return x.view(b, seq_len, -1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query/key/value: (batch, seq_len, d_model)
            mask: optional mask broadcastable to (batch, num_heads, seq_q, seq_k)
        """
        
        # 1. Linear Projections: Apply W_Q, W_K, and W_V to the input.
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. Split Heads: Reshape the projections for multi-head computation.
        Q = self._split_heads(Q)  # (b, h, seq_q, d_k)
        K = self._split_heads(K)  # (b, h, seq_k, d_k)
        V = self._split_heads(V)  # (b, h, seq_k, d_k)

        # 3. Scaled Dot-Product Attention (Q * K^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (b, h, seq_q, seq_k)
        

        # 4. Apply Masking
        if mask is not None:
            # For autoregressive tasks (like decoding), we mask future tokens.
            # We set masked positions (where mask == 0) to a very large negative value.
            scores = scores.masked_fill(mask == 0, float("-1e9"))

        # 5. Softmax and Dropout
        # Softmax converts raw scores into probability distributions (attention weights).
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 6. Compute Context Vector (Attention * V)
        # The attention weights are applied to the Value vectors to produce the weighted sum (context vector) for each position.
        context = torch.matmul(attn, V)  # (b, h, seq_q, d_k)

        # 7. Combine Heads: Reshape the output back to (batch, seq_len, d_model).
        context = self._combine_heads(context)  # (b, seq_q, d_model)
        
        # 8. Final Linear Projection: Apply W_O to the concatenated context vectors.
        out = self.w_o(context)
        
        # Return the final output and the attention weights for possible inspection.
        return out, attn
    

if __name__ == "__main__":
    print("--- Testing MultiHeadAttention ---")
    
    # 1. Setup Configuration
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2
    dropout = 0.0 # Set to 0 for deterministic testing

    # Initialize the module
    mha = MultiHeadAttention(d_model, num_heads, dropout)
    mha.eval() # Switch to eval mode to disable dropout behavior during checks

    # Create dummy input (Batch, Seq_Len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # --- TEST 1: Shape Consistency ---
    output, attn_weights = mha(query=x, key=x, value=x, mask=None)

    print(f"\nTest 1: Shape Checks")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attn weights: {attn_weights.shape}")

    # Assertions
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch!"
    # Attn weights should be (Batch, Num_Heads, Seq_Len, Seq_Len)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), "Attention weights shape mismatch!"
    print("✅ Shape tests passed.")


    # --- TEST 2: Masking Logic (Causal Mask) ---    
    print(f"\nTest 2: Causal Masking Check")
    
    # Create an upper triangular mask
    causal_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len)))
    
    output, attn_weights = mha(x, x, x, mask=causal_mask)
    
    # Check the first batch, first head
    sample_attn = attn_weights[0, 0] # Shape (Seq_Len, Seq_Len)
    
    # The upper triangle (excluding diagonal) should be effectively zero
    upper_triangular = torch.triu(sample_attn, diagonal=1)
    
    # Note: Softmax of -1e9 is not exactly 0.0, but extremely close (e.g., 1e-35)
    if torch.all(upper_triangular < 1e-4):
        print("✅ Causal masking works: Future tokens have 0 probability.")
    else:
        print("❌ Causal masking FAILED: Future tokens are being attended to.")
        print("Upper triangular sum (should be ~0):", upper_triangular.sum().item())


    # --- TEST 3: Padding Mask Logic ---    
    print(f"\nTest 3: Padding Mask Check")
    
    # Mask: 1 for valid tokens, 0 for PAD. 
    # Shape: (Batch, 1, 1, Seq_Len). The '1's allow broadcasting.
    pad_mask = torch.ones((batch_size, 1, 1, seq_len))
    pad_mask[:, :, :, -2:] = 0 # Set last 2 positions to 0 (Masked)
    
    output, attn_weights = mha(x, x, x, mask=pad_mask)
    
    # Check weights for the last 2 columns. They should be 0 for ALL rows.
    last_two_cols = attn_weights[:, :, :, -2:]
    
    if torch.all(last_two_cols < 1e-4):
        print("✅ Padding masking works: [PAD] tokens are ignored.")
    else:
        print("❌ Padding masking FAILED.")

    print("\nAll systems go! 🚀")
import torch
import torch.nn as nn
from typing import Optional

# Import custom submodules
from embeddings import InputEmbeddings
from positional_encoding import PositionalEncoding
from blocks import Encoder, Decoder, ProjectionHead


class Transformer(nn.Module):
    """Transformer model that constructs its own encoder/decoder stacks.

    Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        src_max_len: max source sequence length
        tgt_max_len: max target sequence length
        d_model: model dimension
        num_layers: number of encoder/decoder layers
        num_heads: number of attention heads
        d_ff: feed-forward inner dimension
        dropout: dropout probability
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_max_len: int,
        tgt_max_len: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Embedding Layers ---
        # Input Embeddings map token IDs to d_model vectors.
        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        
        # Positional Encodings inject sequence order information.
        self.src_pos = PositionalEncoding(d_model, max_len=src_max_len, dropout=dropout)
        self.tgt_pos = PositionalEncoding(d_model, max_len=tgt_max_len, dropout=dropout)

        # --- Core Stacks ---
        # Encoder: Stack of N EncoderLayers processing the source sequence.
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        # Decoder: Stack of N DecoderLayers processing the target sequence and encoder output.
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # --- Output Layer ---
        # Projection Head: Maps the final decoder output (d_model) to the vocabulary size.
        self.projection = ProjectionHead(d_model, tgt_vocab_size)

        # --- Parameter Initialization (Xavier/Glorot Uniform) ---
        # Standard practice to stabilize training, especially for large linear layers.
        for p in self.parameters():
            if p.dim() > 1:
                # Applies Xavier initialization to weights (matrices, not biases)
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source tokens.

        src: (batch, src_seq_len) token ids
        src_mask: optional mask broadcastable to attention scores
        returns: (batch, src_seq_len, d_model) encoded representation
        """
        # 1. Embed and Scale
        x = self.src_embed(src)             # (b, seq, d_model)
        # 2. Add Positional Encoding and Dropout
        x = self.src_pos(x)
        # 3. Pass through the Encoder stack
        return self.encoder(x, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decode using encoder outputs and target tokens.

        encoder_output: (batch, src_seq_len, d_model) output from the encoder
        tgt: (batch, tgt_seq_len) target token ids (teacher forcing or partial sequence)
        tgt_mask: causal mask for target self-attention
        returns: (batch, tgt_seq_len, d_model) decoded representation
        """
        # 1. Embed and Scale
        x = self.tgt_embed(tgt)
        # 2. Add Positional Encoding and Dropout
        x = self.tgt_pos(x)
        # 3. Pass through the Decoder stack
        # Requires encoder_output, src_mask (for cross-attention), and tgt_mask (for self-attention)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience forward for training: returns log-probs over target vocab.

        src: (batch, src_seq_len)
        tgt: (batch, tgt_seq_len)
        returns: (batch, tgt_seq_len, vocab_size) log-probabilities
        """
        # 1. Encode the source sequence
        enc = self.encode(src, src_mask)
        
        # 2. Decode the target sequence using the encoder output
        dec = self.decode(enc, src_mask, tgt, tgt_mask)
        
        # 3. Project the final decoder output to vocabulary logits (log-probabilities)
        return self.projection(dec)


if __name__ == "__main__":
    print("--- Testing Full Transformer Model ---")

    # 1. Setup Configuration
    # We use small numbers to keep the test fast and readable
    src_vocab_size = 100
    tgt_vocab_size = 150
    src_max_len = 50
    tgt_max_len = 50
    
    d_model = 512
    num_layers = 2  # Keep it shallow for testing
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12 # Different from src_len to test cross-attention shapes

    # 2. Initialize Model
    model = Transformer(
        src_vocab_size, tgt_vocab_size, 
        src_max_len, tgt_max_len, 
        d_model, num_layers, num_heads, d_ff, dropout
    )
    model.eval() # Set to eval to disable dropout for shape checks
    print("✅ Model initialized successfully.")

    # 3. Create Dummy Data (Random Token IDs)
    # Source: (Batch, Src_Seq_Len)
    src_data = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    # Target: (Batch, Tgt_Seq_Len)
    tgt_data = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # Create Dummy Masks
    # Src Mask: (Batch, 1, 1, Src_Len) - masking nothing for this test
    src_mask = torch.ones((batch_size, 1, 1, src_seq_len))
    # Tgt Mask: (Batch, 1, Tgt_Len, Tgt_Len) - masking nothing for this test
    tgt_mask = torch.ones((batch_size, 1, tgt_seq_len, tgt_seq_len))

    print(f"Source Input Shape: {src_data.shape}")
    print(f"Target Input Shape: {tgt_data.shape}")

    # --- TEST 1: The Encode Method ---
    print("\n--- Test 1: Encoder Pass ---")
    encoder_output = model.encode(src_data, src_mask)
    
    print(f"Encoder Output: {encoder_output.shape}")
    
    # Expected: (Batch, Src_Len, d_model)
    assert encoder_output.shape == (batch_size, src_seq_len, d_model)
    print("✅ Encoder output shape is correct.")


    # --- TEST 2: The Decode Method ---
    print("\n--- Test 2: Decoder Pass ---")
    # Decoder needs the encoder output to perform Cross-Attention
    decoder_output = model.decode(encoder_output, src_mask, tgt_data, tgt_mask)
    
    print(f"Decoder Output: {decoder_output.shape}")
    
    # Expected: (Batch, Tgt_Len, d_model) -- Note: Matches TARGET length
    assert decoder_output.shape == (batch_size, tgt_seq_len, d_model)
    print("✅ Decoder output shape is correct.")


    # --- TEST 3: Full Forward Pass (Training Scenario) ---
    print("\n--- Test 3: Full Forward Pass ---")
    # This runs encode -> decode -> projection
    final_output = model(src_data, tgt_data, src_mask, tgt_mask)
    
    print(f"Final Logits Shape: {final_output.shape}")
    
    # Expected: (Batch, Tgt_Len, Tgt_Vocab_Size)
    expected_shape = (batch_size, tgt_seq_len, tgt_vocab_size)
    assert final_output.shape == expected_shape, \
        f"Expected {expected_shape}, got {final_output.shape}"
    
    # Check if output is valid log-probs (negative values, usually)
    # because log(probability < 1) is negative.
    if final_output.max() <= 0:
        print("✅ Final output range looks like log_softmax (values <= 0).")
    else:
        print("⚠️ Warning: Found positive values. Check if ProjectionHead uses log_softmax.")

    print("\nTransformer Integration Test Passed! 🚀")
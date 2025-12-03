from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def causal_mask(size: int):
    """
    Args:
        size (int): The sequence length.

    Returns:
        torch.Tensor: A (1, size, size) boolean mask.
    """
    # Create an upper triangular matrix with ones, everything else is zero
    # diagonal=1 means the main diagonal is also set to zero
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    """
    A custom PyTorch Dataset for handling bilingual text data for a Transformer model.
    It takes raw text pairs, tokenizes them, adds special tokens, pads them to a
    fixed sequence length, and creates the necessary attention masks.
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        """
        Args:
            ds (Dataset): The raw dataset (e.g., from Hugging Face).
            tokenizer_src (Tokenizer): The tokenizer for the source language.
            tokenizer_tgt (Tokenizer): The tokenizer for the target language.
            seq_len (int): The fixed sequence length for all tensors.
        """
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len


        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)


    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.ds)
    
    def __getitem__(self, index: int) -> Any:
        """
        Retrieves and preprocesses a single data pair (source, target) at the given index.
        This method is called by the DataLoader to create a batch.
        """
        # Step 1: Get the raw text pair from the original dataset for the given index
        src_target_pair = self.ds[index]
        # The dataset structure assumes 'translation' is a dictionary with language keys
        # Ensure your raw dataset provides pairs like: {'translation': {'en': '...', 'it': '...'}} from {'id': index, 'translation': {'en': '...', 'fr': '...'}}
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Step 2: Tokenize the text into numerical IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Step 3: Calculate how much padding is needed for each sequence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # Encoder needs space for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Decoder input only needs space for [SOS] at the beginning
        # The label (decoder output) needs space for [EOS] at the end
        label_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # The label has [EOS] while decoder input has [SOS].

        # Step 4: Check if the sentences are too long. If so, they cannot be used.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence length exceeds the maximum sequence length.")
        
        # Step 5: Build the final tensors for the model

        # (5a) Encoder Input: [SOS] + Source Sentence + [EOS] + Padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # (5b) Decoder Input: [SOS] + Target Sentence + Padding
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # (5c) Label (what the model should predict): Target Sentence + [EOS] + Padding
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * label_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # Step 6: Sanity checks on final length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
    
        # Step 7: Create attention masks
        # The dictionary format is what the DataLoader will collate into a batch.
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # shape: (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # decoder output
            "src_text": src_text,
            "tgt_text": tgt_text,
       }
    
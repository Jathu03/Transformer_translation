# Handles downloading the dataset, splitting it, and creating PyTorch DataLoaders.

from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from dataset import BilingualDataset  # Imported from your existing dataset.py
from tokenizer_utils import get_or_build_tokenizer

def get_ds(config):
    """
    Loads the dataset, builds tokenizers, and returns dataloaders.
    """
    # Load raw dataset from Hugging Face
    ds_raw = load_dataset(
        'opus_books', 
        f"{config['lang_src']}-{config['lang_tgt']}", 
        split='train'
    )

    # Build or Load Tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split Train/Validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create Pytorch Datasets
    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    # Calculate max lengths (for information only)
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    # Create DataLoaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # Validation batch size is usually 1 for translation tasks to evaluate sentence by sentence
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True) 

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



if __name__ == "__main__":
    import sys
    
    # 1. Define a Test Configuration
    # We use a small sequence length and batch size for quick inspection
    test_config = {
        'lang_src': 'en',
        'lang_tgt': 'it',
        'seq_len': 350,           # Keep it small for display
        'batch_size': 4,         # Small batch size
        'tokenizer_file': 'tokenizer_{0}.json' # Pattern used by tokenizer_utils
    }

    print("--- Testing Data Loader ---")
    print(f"Configuration: {test_config}")

    try:
        # 2. Run the function
        # Note: This will trigger a download of opus_books if not cached (~50MB)
        # It will also train tokenizers if .json files don't exist
        print("\n[1/3] Calling get_ds()... (This might take time on first run)")
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(test_config)

        # 3. Inspect Tokenizers
        print("\n[2/3] Inspecting Tokenizers...")
        src_vocab_size = tokenizer_src.get_vocab_size()
        tgt_vocab_size = tokenizer_tgt.get_vocab_size()
        print(f"Source Vocab Size: {src_vocab_size}")
        print(f"Target Vocab Size: {tgt_vocab_size}")
        
        assert src_vocab_size > 0, "Source tokenizer is empty!"
        assert tgt_vocab_size > 0, "Target tokenizer is empty!"

        # 4. Inspect DataLoaders and Batches
        print("\n[3/3] Inspecting DataLoaders...")
        
        # Get one batch from the training dataloader
        train_batch = next(iter(train_dataloader))
        
        # Keys expected from BilingualDataset.__getitem__
        expected_keys = ['encoder_input', 'decoder_input', 'encoder_mask', 'decoder_mask', 'label', 'src_text', 'tgt_text']
        
        print("Batch keys present:", list(train_batch.keys()))
        
        # Check Shapes
        # Encoder Input: (Batch_Size, Seq_Len)
        enc_input = train_batch['encoder_input']
        assert enc_input.shape == (test_config['batch_size'], test_config['seq_len']), \
            f"Encoder Input shape mismatch. Got {enc_input.shape}"
        
        # Encoder Mask: (Batch_Size, 1, 1, Seq_Len)
        enc_mask = train_batch['encoder_mask']
        assert enc_mask.shape == (test_config['batch_size'], 1, 1, test_config['seq_len']), \
            f"Encoder Mask shape mismatch. Got {enc_mask.shape}"

        # Decoder Mask: (Batch_Size, 1, Seq_Len, Seq_Len)
        dec_mask = train_batch['decoder_mask']
        assert dec_mask.shape == (test_config['batch_size'], 1, test_config['seq_len'], test_config['seq_len']), \
            f"Decoder Mask shape mismatch. Got {dec_mask.shape}"

        print("✅ Batch shapes are correct.")
        
        # Check Validation Loader (Should have batch_size = 1)
        val_batch = next(iter(val_dataloader))
        val_input = val_batch['encoder_input']
        assert val_input.shape[0] == 1, f"Validation batch size should be 1, got {val_input.shape[0]}"
        print("✅ Validation loader has batch_size=1.")

        print("\nTest Passed! Data Loader is working correctly. 🚀")

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
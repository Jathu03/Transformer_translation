import warnings
import torch
import torch.nn as nn
from pathlib import Path

from src.transformer_mt.config import get_config


from src.transformer_mt.data_loader import get_ds
from src.transformer_mt.trainer import Trainer
from src.transformer_mt.model_architecture.transformer import Transformer

def get_model(config, src_vocab_size, tgt_vocab_size):
    """
    Initializes the Transformer model using parameters from the config.
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_max_len=config['seq_len'],
        tgt_max_len=config['seq_len'],
        d_model=config.get('d_model', 512),
        num_layers=config.get('num_layers', 6),  # Default to 6 if not in config
        num_heads=config.get('num_heads', 8),    # Default to 8 if not in config
        d_ff=config.get('d_ff', 2048),           # Default to 2048 if not in config
        dropout=config.get('dropout', 0.1)
    )
    return model

def main():
    # 1. Load Config
    config = get_config()
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model weights folder if it doesn't exist
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # 2. Data Loading
    # This handles downloading, tokenizing, and creating dataloaders
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # 3. Model Initialization
    model = get_model(
        config, 
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # CrossEntropyLoss with Label Smoothing
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        label_smoothing=0.1
    ).to(device)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        optimizer=optimizer,
        criterion=loss_fn,
        config=config,
        device=device
    )

    # 6. Start Training
    # Handles preloading checkpoint if specified in config
    trainer.preload_model()
    trainer.train()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
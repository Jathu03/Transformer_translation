# Handles the creation, training, and loading of tokenizers.

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

def get_all_sentences(ds, lang):
    """
    Generator to efficiently yield sentences for tokenizer training.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Builds a WordLevel tokenizer if it doesn't exist, or loads it if it does.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer for language '{lang}' not found. Building a new one...")
        # Initialize
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Trainer configuration
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=2
        )
        
        # Train
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        # Save
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading existing tokenizer for language '{lang}' from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer
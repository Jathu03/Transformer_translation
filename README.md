# Transformer Translation Model

## Description

This project implements a Transformer-based neural machine translation model from scratch using PyTorch. It is designed for translating text between languages, with a default setup for English to Italian translation using the Opus Books dataset. The model follows the architecture described in the paper "Attention is All You Need" by Vaswani et al., including multi-head attention, positional encoding, and encoder-decoder stacks.

## Features

- **Custom Transformer Architecture**: Modular implementation with separate encoder and decoder stacks.
- **Word-Level Tokenization**: Uses Hugging Face's tokenizers library for efficient text processing.
- **Training and Validation**: Supports training with teacher forcing, validation, and checkpoint saving/resuming.
- **Logging**: Integrated TensorBoard logging for monitoring training progress.
- **Configurable**: Easily adjustable hyperparameters via a configuration file.
- **Dataset Handling**: Loads and preprocesses bilingual datasets from Hugging Face.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <your-repo-url>
   cd Github_Repo
   ```

2. **Install Dependencies**:

   - Install PyTorch (adjust for your CUDA version if using GPU):
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - Install other required packages:
     ```bash
     pip install datasets tokenizers tqdm tensorboard
     ```

   Note: Ensure you have Python 3.7+ installed.

## Usage

### Training the Model

1. **Configure the Model**:

   - Edit `config.py` to set hyperparameters such as batch size, number of epochs, source/target languages, sequence length, and model dimensions.

2. **Run Training**:
   ```bash
   python train.py
   ```
   - This will download the dataset, build tokenizers (if not present), and start training.
   - Checkpoints are saved in the `weights/` directory.
   - Training logs are available via TensorBoard: `tensorboard --logdir runs/model`.

### Inference

The project currently focuses on training. For inference, you can extend the code by implementing a greedy decoding or beam search function in `trainer.py` or a separate script. Example usage would involve loading a trained model and generating translations for input sentences.

### Configuration

Key settings in `config.py`:

- `batch_size`: Number of samples per batch.
- `num_epochs`: Total training epochs.
- `lr`: Learning rate.
- `seq_len`: Maximum sequence length.
- `d_model`: Model dimension.
- `lang_src` and `lang_tgt`: Source and target languages (e.g., 'en' and 'it').
- `model_folder`: Directory for saving model weights.
- `preload`: Epoch to resume training from (set to None for fresh start).

## Project Structure

```
Transformer_Translation/
├── config.py                 # Configuration settings
├── dataset.py                # BilingualDataset class for data preprocessing
├── data_loader.py            # Data loading, tokenization, and DataLoader creation
├── tokenizer_utils.py        # Tokenizer building and loading utilities
├── train.py                  # Main training script
├── trainer.py                # Trainer class handling training logic and logging
├── model_architecture/       # Transformer model components
│   ├── __init__.py
│   ├── transformer.py        # Main Transformer model class
│   ├── blocks.py             # Encoder/Decoder layers and residual connections
│   ├── attention.py          # Multi-head attention mechanism
│   ├── embeddings.py         # Input embeddings
│   ├── feedforward.py        # Feed-forward network
│   └── positional_encoding.py # Positional encoding
├── weights/                  # Saved model checkpoints (generated during training)
├── runs/                     # TensorBoard logs (generated during training)
└── README.md                 # This file
```

## Dependencies

- `torch`: PyTorch for deep learning.
- `datasets`: Hugging Face library for dataset loading.
- `tokenizers`: For building and using tokenizers.
- `tqdm`: Progress bars for training loops.
- `tensorboard`: For logging and visualization.

## How It Works

1. **Data Preparation**: The `data_loader.py` script loads the Opus Books dataset, builds word-level tokenizers for source and target languages, and creates PyTorch DataLoaders.
2. **Model Architecture**: The Transformer consists of an encoder (processes source text) and a decoder (generates target text using encoder outputs). Key components include multi-head attention, feed-forward networks, and positional encodings.
3. **Training**: The `trainer.py` class manages the training loop, loss computation (Cross-Entropy with label smoothing), backpropagation, and validation.
4. **Tokenization**: Special tokens like [SOS], [EOS], [PAD], and [UNK] are used for sequence handling.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Based on the Transformer architecture from "Attention is All You Need" (Vaswani et al., 2017).
- Uses datasets and tokenizers from Hugging Face.

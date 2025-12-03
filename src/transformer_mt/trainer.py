# Encapsulates the training logic, state management, logging, and checkpoint saving into a class.

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from config import get_weights_file_path

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, optimizer, criterion, config, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir=config['experiment_name'])
        self.global_step = 0
        self.initial_epoch = 0

    def preload_model(self):
        """Resumes training if a preload epoch is specified in config."""
        if self.config['preload']:
            model_filename = get_weights_file_path(self.config, self.config['preload'])
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            self.initial_epoch = state['epoch'] + 1
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.global_step = state['global_step']
            print(f"Resumed from global_step {self.global_step}")

    def train_epoch(self, epoch_index):
        self.model.train()
        batch_iterator = tqdm(self.train_dataloader, desc=f"Processing epoch {epoch_index:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(self.device) # (B, Seq_len)
            decoder_input = batch['decoder_input'].to(self.device) # (B, Seq_len)
            encoder_mask = batch['encoder_mask'].to(self.device)   # (B, 1, 1, Seq_len)
            decoder_mask = batch['decoder_mask'].to(self.device)   # (B, 1, Seq_len, Seq_len)
            label = batch['label'].to(self.device)                 # (B, Seq_len)

            # Run the tensors through the transformer
            encoder_output = self.model.encode(encoder_input, encoder_mask) 
            decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = self.model.project(decoder_output) # (B, Seq_len, tgt_vocab_size)

            # Calculate loss
            # (B, Seq_len, tgt_vocab_size) -> (B * Seq_len, tgt_vocab_size)
            loss = self.criterion(
                proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), 
                label.view(-1)
            )
            
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log to Tensorboard
            self.writer.add_scalar('train_loss', loss.item(), self.global_step)
            self.writer.flush()

            # Backprop
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1

    def validate(self):
        """
        Runs validation. (Basic implementation - can be expanded to calculate BLEU)
        """
        self.model.eval()
        count = 0
        
        # Example: Check the first 2 sentences to print to console
        # (Full validation logic typically calculates BLEU score here)
        with torch.no_grad():
            for batch in self.val_dataloader:
                count += 1
                encoder_input = batch['encoder_input'].to(self.device)
                encoder_mask = batch['encoder_mask'].to(self.device)
                
                # Get the model output (Greedy Decode)
                # Note: You need a greedy_decode function in your dataset/utils or imported
                # For now, we just skip full decoding to keep the file clean, 
                # but this is where you would compare src -> model_out vs tgt
                if count == 2: break

    def save_checkpoint(self, epoch):
        model_filename = get_weights_file_path(self.config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }, model_filename)

    def train(self):
        for epoch in range(self.initial_epoch, self.config['num_epochs']):
            self.train_epoch(epoch)
            self.validate() # Optional: Run validation
            self.save_checkpoint(epoch)
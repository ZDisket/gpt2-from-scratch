#!/usr/bin/env python3
import argparse
import os
import time
import math
import glob
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

import tiktoken
import wandb
from tqdm import tqdm  # Import tqdm


# -----------------------------
# Configuration and Hyperparameters
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT-2-like Language Model with AMP")

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/', 
                        help='Path to the folder containing training text files')

    # Model hyperparameters (Defaults set to GPT-2-124M)
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=768, help='Embedding size')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--context_length', type=int, default=1024, help='Maximum context length')

    # Training hyperparameters
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    # INOP
    parser.add_argument('--eval_steps', type=int, default=99999999999999999999999, help='Evaluate every n steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay for optimizer')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Train/Eval split ratio')



    # INOP
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='Number of steps to accumulate gradients before updating the model')
    
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda or cpu)')

    # Weights & Biases
    parser.add_argument('--wandb_project', type=str, default='gpt2-training', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team name')

    # Test Interval and Prompts
    parser.add_argument('--test_interval', type=int, default=1000, help='Interval (in steps) to test the model')
    parser.add_argument('--test_prompts', type=str, nargs='+', default=["Once upon a time"],
                        help='Prompts to use for testing the model')

    # AMP
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--amp_dtype', type=str, choices=['float16', 'bfloat16'], default='bfloat16',
                        help='AMP precision type. Choose "bfloat16" if supported by the device.')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


# -----------------------------
# Dataset and DataLoader
# -----------------------------


class TextDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer, context_length: int):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.tokens = []
        self._prepare_tokens(file_paths)

    def _prepare_tokens(self, file_paths: List[str]):
        for file_path in file_paths:
            # Create the corresponding .pt file name
            pt_file_path = file_path + '.pt'
            
            # Check if the .pt file exists
            if os.path.exists(pt_file_path):
                print(f"Loading tokenized data from {pt_file_path}...")
                # Load the tokenized data from the .pt file
                tokens = torch.load(pt_file_path)
            else:
                print(f"Tokenizing data from {file_path}...")
                # Tokenize the data and store it
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    tokens = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
                
                # Save the tokenized data to the .pt file
                print(f"Saving tokenized data to {pt_file_path}...")
                torch.save(tokens, pt_file_path)
            
            # Extend the tokens list with the newly loaded/saved tokens
            print("Extending with tokens...")
            self.tokens.extend(tokens)
        
        # Convert the final token list into a tensor
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens)}")

    def __len__(self):
        return (len(self.tokens) - self.context_length) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]
        return x, y


# -----------------------------
# Model Definitions
# -----------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.size()

        # Linear projections
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal Masking
        mask = torch.triu(torch.ones((seq_length, seq_length), device=inputs.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)  # (B, n_heads, seq_length, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, seq_length, d_model)

        out = self.fc_out(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, context_length: int, d_model: int):
        super().__init__()
        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1), :]


@torch.jit.script
def relugt_forward(x, slope, alpha_neg, alpha_pos):
    return torch.where(x < 0, alpha_neg * slope * x, alpha_pos * x ** 2)

class ReLUGT(nn.Module):
    """
    ReLU GT: Leaky squared ReLU with trainable positive alpha, slope, and static negative alpha.
    Early experiments show near parity with APTx S1 with faster initial fitting. Only squares positive part.

    5x faster when forward pass is torch.jit.script
    """
    def __init__(self, initial_slope=0.05, initial_alpha_neg=2.5, initial_alpha_pos=1.0):
        super(ReLUGT, self).__init__()
        self.slope = nn.Parameter(torch.tensor(initial_slope))
        self.alpha_neg = nn.Parameter(torch.tensor(initial_alpha_neg))  # Changed to tensor
        self.alpha_pos = nn.Parameter(torch.tensor(initial_alpha_pos))

    def forward(self, x):
        return relugt_forward(x, self.slope, self.alpha_neg, self.alpha_pos)


# taken & expanded from https://github.com/cloneofsimo/minRF/blob/4fc10e0cc8ba976152c7936a1af9717209f03e18/dit.py#L140
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None, act="swiglu"):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        if act == "swiglu":
            self.act_fn = self._forward_silu_gating
        elif act == "relugtz":
            self.relugt = ReLUGT()
            self.act_fn = self._forward_relugtz_gating
        elif act == "geglu":
            self.act_fn = self._forward_gelu_gating
        else:
            raise RuntimeError(f"Unknown activation function {act}")

    def _forward_gelu_gating(self, x1, x3):
        return F.gelu(x1) * x3

    def _forward_relugtz_gating(self, x1, x3):
        return self.relugt(x1) * x3

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x), self.w3(x)))

class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, 4 * d_model, 192, act="swiglu")
        
        #self.ffn = nn.Sequential(
        #    nn.Linear(d_model, 4 * d_model),
         #   nn.GELU(),
          #  nn.Linear(4 * d_model, d_model),
           # nn.Dropout(dropout)
        #)
    
    def forward(self, x: torch.Tensor):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, context_length: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(context_length, d_model)
        self.layers = nn.ModuleList([GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Tie weights
        self.fc_out.weight = self.embedding.weight
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        x = self.embedding(x)  # (B, T, C)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.fc_out(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, x: torch.Tensor, max_new_tokens: int, context_length: int):
        for _ in range(max_new_tokens):
            if x.size(1) > context_length:
                x = x[:, -context_length:]
            logits, _ = self.forward(x)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x


# -----------------------------
# Training and Evaluation
# -----------------------------

def train(model, optimizer, scheduler, scaler, train_loader, eval_loader, args):
    model.train()
    step = 0
    total_steps = args.epochs * len(train_loader)
    progress_bar = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(1, args.epochs + 1):
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
        for xb, yb in epoch_iterator:
            xb, yb = xb.to(args.device), yb.to(args.device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=torch.bfloat16 if (args.use_amp and args.amp_dtype == 'bfloat16') else torch.float16):
                logits, loss = model(xb, yb)
            
            # Scale loss and backward
            if args.use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            
            scheduler.step()
            
            step += 1
            wandb.log({'train_loss': loss.item(), 'step': step})
            
            # Update progress bars
            epoch_iterator.set_postfix({'loss': loss.item()})
            progress_bar.update(1)
            
            # Evaluation
            if step % args.eval_steps == 0:
                eval_loss = evaluate(model, eval_loader, args)
                wandb.log({'eval_loss': eval_loss, 'step': step})
                progress_bar.set_postfix({'current_step': step, 'train_loss': loss.item(), 'eval_loss': eval_loss})
                print(f"Step: {step} | Train Loss: {loss.item():.4f} | Eval Loss: {eval_loss:.4f}")
            
            # Testing
            if step % args.test_interval == 0:
                test_results = test_model(model, args)
                for prompt, generated in test_results.items():
                    print(f"\n=== Test Prompt: \"{prompt}\" ===")
                    print(generated)
                    wandb.log({f"generated_text_{prompt}": generated, 'step': step})

    progress_bar.close()



def evaluate(model, eval_loader, args):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=torch.bfloat16 if (args.use_amp and args.amp_dtype == 'bfloat16') else torch.float16):
                _, loss = model(xb, yb)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count if count > 0 else float('inf')


def test_model(model, args):
    model.eval()
    test_results = {}
    with torch.no_grad():
        for prompt in args.test_prompts:
            input_ids = torch.tensor(args.tokenizer.encode(prompt), dtype=torch.long, device=args.device).unsqueeze(0)
            generated_ids = model.generate(input_ids, max_new_tokens=100, context_length=args.context_length)
            generated_text = args.tokenizer.decode(generated_ids[0].tolist())
            test_results[prompt] = generated_text
    model.train()
    return test_results


# -----------------------------
# Main Function
# -----------------------------

def main():
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Initialize W&B
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    assert tokenizer.n_vocab == args.vocab_size, "Tokenizer vocab size does not match the specified vocab_size."
    args.tokenizer = tokenizer  # Assign tokenizer to args for access in test_model

    # Prepare data
    file_paths = glob.glob(os.path.join(args.data_dir, '*.txt'))
    if not file_paths:
        raise ValueError(f"No .txt files found in {args.data_dir}")

    dataset = TextDataset(file_paths, tokenizer, args.context_length)
    n_train = int(len(dataset) * args.split_ratio)
    n_eval = len(dataset) - n_train
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [n_train, n_eval])

    train_loader = TorchDataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    eval_loader = TorchDataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True)

    # Initialize model
    model = GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_length=args.context_length
    ).to(args.device)


    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=args.lr * 0.1)

    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Start training
    start_time = time.time()
    train(model, optimizer, scheduler, scaler, train_loader, eval_loader, args)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    wandb.save("final_model.pt")

    # Generate sample text
    prompt = "Once upon a time"
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=args.device).unsqueeze(0)
    generated_ids = model.generate(input_ids, max_new_tokens=500, context_length=args.context_length)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print("\nGenerated Text:\n")
    print(generated_text)

    # Log final generated text
    wandb.log({"final_generated_text": generated_text})

    wandb.finish()


if __name__ == "__main__":
    main()

"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import csv
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset,collate_fn
import matplotlib.pyplot as plt
import seaborn as sns

import os
import math
import numpy as np
import random
import logging
import argparse
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu
import wandb


##Loggr setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

##Set seed
def set_seed(seed: int):
    """
    Setting random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
#        raise NotImplementedError
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
##        raise NotImplementedError
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)

            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1,target.unsqueeze(1),self.confidence)

            true_dist[:, self.pad_idx] = 0
            mask = (target == self.pad_idx)
            true_dist[mask] = 0

        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        return loss

def log_attention_maps(model, sentence_tokens, epoch):

    last_encoder_layer = model.encoder.layers[-1]

    attn = last_encoder_layer.self_attn.attention_weights

    if attn is None:
        return

    attn = attn[0]  # first sample

    num_heads = attn.shape[0]

    for h in range(num_heads):

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            attn[h].numpy(),
            xticklabels=sentence_tokens,
            yticklabels=sentence_tokens,
        )

        plt.title(f"Head {h}")

        if wandb.run is not None:
            wandb.log({
                f"attention_head_{h}":
                    wandb.Image(plt)
            })

        plt.close()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    ##raise NotImplementedError
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(data_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        # Decoder input and target
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt_input).to(device)

        if is_train:
            optimizer.zero_grad()

        ##Raw output(logits) from model
        logits = model(src,tgt_input,src_mask,tgt_mask)

        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        probs = torch.softmax(logits, dim=-1)

        valid_mask = tgt_output != 1
        
        correct_probs = probs[
            torch.arange(probs.size(0)),
            tgt_output
        ]
        
        prediction_confidence = correct_probs[valid_mask].mean().item()
        if wandb.run is not None:
            wandb.log({
                "prediction_confidence": prediction_confidence
            })

        loss = loss_fn(logits, tgt_output)
        if is_train:
            loss.backward()
            
            if (
                is_train
                and scheduler is not None
                and scheduler.last_epoch < 1000
            ):
            
                q_grad = 0.0
                k_grad = 0.0
            
                for name, param in model.named_parameters():
            
                    if param.grad is None:
                        continue
            
                    if "W_q.weight" in name:
                        q_grad += param.grad.norm().item()
            
                    if "W_k.weight" in name:
                        k_grad += param.grad.norm().item()
            
                if wandb.run is not None:
                    wandb.log({
                        "query_grad_norm": q_grad,
                        "key_grad_norm": k_grad,
                        "step": scheduler.last_epoch
                    })

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            phase = "TRAIN" if is_train else "VAL"

            logger.info(
                f"{phase} | "
            #logger.info(
                f"Epoch [{epoch_num}] "
                f"Batch [{batch_idx}/{len(data_iter)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(data_iter)
    return avg_loss
    


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
##    raise NotImplementedError
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys).to(device)
        out = model.decode(memory,src_mask,ys,tgt_mask)
        prob = out[:, -1]

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        next_tensor = torch.ones(1, 1).type_as(src.data).fill_(next_word)

        ys = torch.cat([ys, next_tensor.to(device)],dim=1)
        if next_word == end_symbol:
            break
    return ys

# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    #raise NotImplementedError
    model.eval()

    references = []
    hypotheses = []

    #sos_idx = tgt_vocab["<sos>"]
    #eos_idx = tgt_vocab["<eos>"]
    #pad_idx = tgt_vocab["<pad>"]

    #idx_to_token = {v: k for k, v in tgt_vocab.items()}
    
    sos_idx = tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.stoi["<eos>"]
    pad_idx = tgt_vocab.stoi["<pad>"]
    
    idx_to_token = tgt_vocab.itos
    
    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            src_mask = make_src_mask(src).to(device)

            for i in range(src.size(0)):
                src_single = src[i].unsqueeze(0)
                src_mask_single = src_mask[i].unsqueeze(0)
                pred_tokens = greedy_decode(model,src_single,src_mask_single,max_len=max_len,start_symbol=sos_idx,end_symbol=eos_idx,device=device)
                pred_sentence = []
                for idx in pred_tokens.squeeze().tolist():
                    if idx in [sos_idx, eos_idx, pad_idx]:
                        continue
                    pred_sentence.append(idx_to_token[idx])
                
                target_sentence = []
                for idx in tgt[i].tolist():
                    if idx in [sos_idx, eos_idx, pad_idx]:
                        continue
                    target_sentence.append(idx_to_token[idx])

                hypotheses.append(pred_sentence)
                references.append([target_sentence])

    bleu = corpus_bleu(references, hypotheses) * 100
    return bleu


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    src_vocab,
    tgt_vocab,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    #raise NotImplementedError
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None else None,

        "model_config": model.config,

        # IMPORTANT
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved at {path}")

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer=None,
    scheduler=None,
):
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    #raise NotImplementedError
    checkpoint = torch.load(path, map_location="cpu",weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    logger.info(f"Checkpoint loaded from {path}")

    return (
        checkpoint["epoch"],
        checkpoint.get("src_vocab"),
        checkpoint.get("tgt_vocab"),
    )


##CLI ARGUMENTSS

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=4000)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_path", type=str, default="OP/baseline.pt")
    parser.add_argument("--save_path_last", type=str, default="OP/Last_checkpoint.pt")

    # W&B
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="DA6401_Assignment3")
    parser.add_argument("--wandb_task_name", type=str, default="Transformer")
    
    # Scheduler experiments
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="noam",
        choices=["noam", "fixed"]
    )
    
    # Attention scaling ablation
    parser.add_argument(
        "--use_scaling",
        type=int,
        default=1
    )
    
    # Positional encoding experiment
    parser.add_argument(
        "--pos_encoding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learned"]
    )
    
    # Label smoothing experiment
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1
    )
    
    # Attention visualization
    parser.add_argument(
        "--save_attention_maps",
        type=int,
        default=0
    )
    
    # Optional experiment naming
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline"
    )

    return parser.parse_args()

# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    #raise NotImplementedError
    args = get_args()

    logger.info(f"Arguments: {args}")

    set_seed(args.seed)
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path_last), exist_ok=True)
    

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Using device: {device}")

    # ==========================================================
    # W&B SETUP
    # ==========================================================

    use_wandb = args.wandb_api_key is not None

    if use_wandb:

        os.environ["WANDB_API_KEY"] = args.wandb_api_key

        run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.wandb_task_name}_"
                 f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
        )

    else:

        run = None

        logger.info("Training WITHOUT W&B")

    ## Dataset loading and dataloader

    logger.info("Loading datasets...")

    train_dataset = Multi30kDataset(split="train")
    
    val_dataset = Multi30kDataset(
        split="validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )
    
    test_dataset = Multi30kDataset(
        split="test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )
    
    src_vocab_size = len(train_dataset.src_vocab)
    tgt_vocab_size = len(train_dataset.tgt_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    ##Model Init
    logger.info("Building Transformer model...")
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_scaling=bool(args.use_scaling),
        pos_encoding=args.pos_encoding,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4 if args.scheduler_type == "fixed" else args.lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    if args.scheduler_type == "noam":
        scheduler = NoamScheduler(
            optimizer,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps
        )
    else:
        scheduler = None

    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=1,
#        smoothing=0.1
        smoothing=args.label_smoothing
    )

    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/{args.experiment_name}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "bleu",
        "learning_rate"
    ])

    ##Training Loop
    best_val_loss = float("inf")
    
    best_bleu = -1
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Starting Epoch {epoch}")
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device
        )
        
        logger.info("Starting BLEU evaluation...")
        
        val_bleu = evaluate_bleu(
            model,
            val_loader,
            train_dataset.tgt_vocab,
            device=device
        )
        
        logger.info(f"BLEU completed: {val_bleu}")
        
        if args.save_attention_maps:
            sample_src, _ = next(iter(test_loader))
            sample_src = sample_src.to(device)
            sample_mask = make_src_mask(sample_src).to(device)
            with torch.no_grad():
                model.encode(sample_src, sample_mask)
            tokens = [
                train_dataset.src_vocab.itos[idx.item()]
                for idx in sample_src[0]
            ]
        
            log_attention_maps(model, tokens, epoch)
        
        csv_writer.writerow([
            epoch,
            train_loss,
            val_loss,
            val_bleu,
            optimizer.param_groups[0]["lr"]
        ])
        
        csv_file.flush()
        
        logger.info(
            f"Epoch [{epoch}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
            f"val_bleu: {val_bleu:.4f}" ,
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bleu": val_bleu,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
        
        improved = False
        if val_bleu > best_bleu:
            improved = True
        
        elif val_bleu == best_bleu and val_loss < best_val_loss:
            improved = True
        
        if improved:
            best_bleu = val_bleu
            best_val_loss = val_loss
            patience_counter = 0
        
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                train_dataset.src_vocab,
                train_dataset.tgt_vocab,
                args.save_path
            )
        
            logger.info(f"Best model saved at epoch {epoch}")
        
        else:
            patience_counter += 1
            logger.info(
                f"No improvement for {patience_counter} epoch(s)"
            )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            train_dataset.src_vocab,
            train_dataset.tgt_vocab,
            args.save_path_last
        )
        
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    ##BLEU eval on test data
    logger.info("Evaluating BLEU score...")
    
    load_checkpoint(args.save_path, model)

    bleu = evaluate_bleu(model,test_loader,train_dataset.tgt_vocab,device=device)
    logger.info(f"Final BLEU Score: {bleu:.2f}")
    
    csv_file.close()
    
    if use_wandb:
        wandb.log({"test_bleu": bleu})
        run.finish()
    


if __name__ == "__main__":
    run_training_experiment()

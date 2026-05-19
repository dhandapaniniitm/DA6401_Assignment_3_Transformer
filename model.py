"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling=True,
):

    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))

    if use_scaling:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

    return output, attn_weights

# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    #raise NotImplementedError
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    ##raise NotImplementedError
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device),diagonal=1).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        #raise NotImplementedError
        self.use_scaling = True
        self.attention_weights = None
        
        ##Linear Projections for Query, Key, and Value
        ##Each converts the following [batch, seq_len, d_model] to [batch, seq_len, d_model]
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
       
        ##Final projection after concatenating all heads together
        self.W_o = nn.Linear(d_model, d_model)
       
        ##Dropout being applied to attention outputs
        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    
        batch_size = query.size(0)
    
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
    
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
    
        # Attention
        attn_output, attn_weights = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
            use_scaling=self.use_scaling
        )
    
        # Apply dropout ONLY once
        attn_output = self.dropout(attn_output)
    
        # DO NOT move attention weights to CPU during inference
        self.attention_weights = attn_weights
    
        # Combine heads
        out = self.combine_heads(attn_output)
    
        return self.W_o(out)
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size,seq_len,self.num_heads,self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        #raise NotImplementedError
        
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        ##raise NotImplementedError
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        positions = torch.arange(
            0,
            seq_len,
            device=x.device
        ).unsqueeze(0)

        pos_embed = self.pos_embedding(positions)

        x = x + pos_embed

        return self.dropout(x)

# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)
        ##raise NotImplementedError
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
##        raise NotImplementedError
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO:instantiate:
        #raise NotImplementedError
        self.self_attn = MultiHeadAttention(d_model,num_heads,dropout)
        
        self.ffn = PositionwiseFeedForward(d_model,d_ff,dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        #raise NotImplementedError
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:
        ##raise NotImplementedError
        self.self_attn = MultiHeadAttention(d_model,num_heads,dropout)
        self.cross_attn = MultiHeadAttention(d_model,num_heads,dropout)
        self.ffn = PositionwiseFeedForward(d_model,d_ff,dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        #raise NotImplementedError
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        
        attn = self.cross_attn(x,memory,memory,src_mask)
        
        x = self.norm2(x + self.dropout(attn))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x
        

# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)
#        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        #raise NotImplementedError
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        ##raise NotImplementedError
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        #raise NotImplementedError
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """
    def __init__(
        self,
        #src_vocab_size: int,
        #tgt_vocab_size: int,
        src_vocab_size: int =7853,
        tgt_vocab_size: int =5893,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.3,
        checkpoint_path: str = "baseline.pt",
        use_scaling: bool = True,
        pos_encoding="sinusoidal",
    ) -> None:
        super().__init__()
        # TODO: Instantiate 
        # init should also load the model weights if checkpoint path provided, download the .pth file like this
        if checkpoint_path is not None:
            gdown.download(id="1IVGJhD1u8HCLEbaY3AC3EVBcto7z5gSc", output=checkpoint_path, quiet=False)
        ##raise NotImplementedError
        
        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "use_scaling": use_scaling,
            "pos_encoding": pos_encoding,
        }

        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model)

#        self.positional_encoding = PositionalEncoding(d_model,dropout)
        if pos_encoding == "sinusoidal":
            self.positional_encoding = PositionalEncoding(d_model, dropout)
        else:
            self.positional_encoding = LearnedPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = EncoderLayer(d_model,num_heads,d_ff,dropout)
        decoder_layer = DecoderLayer(d_model,num_heads,d_ff,dropout)
        
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        
        for layer in self.encoder.layers:
            layer.self_attn.use_scaling = use_scaling
        
        for layer in self.decoder.layers:
            layer.self_attn.use_scaling = use_scaling
            layer.cross_attn.use_scaling = use_scaling
        
        #if checkpoint_path is not None:
        #    checkpoint = torch.load(checkpoint_path,map_location="cpu")
        #    self.load_state_dict(checkpoint["model_state_dict"])
            
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False
            )
        
            # Load vocab if present
            if "src_vocab" in checkpoint:
                self.src_vocab = checkpoint["src_vocab"]
        
            if "tgt_vocab" in checkpoint:
                self.tgt_vocab = checkpoint["tgt_vocab"]
        
            # Load model weights
            self.load_state_dict(
                checkpoint["model_state_dict"],
                strict=False
            )

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """  
##        raise NotImplementedError
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        #raise NotImplementedError
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        out = self.decoder(x,memory,src_mask,tgt_mask)
        return self.fc_out(out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
#        raise NotImplementedError
        memory = self.encode(src, src_mask)
        return self.decode(memory,src_mask,tgt,tgt_mask)

    def infer(self, src_sentence: str) -> str:
        self.eval()
        torch.set_grad_enabled(False)
        device = next(self.parameters()).device
    
        if not hasattr(self, "src_vocab") or not hasattr(self, "tgt_vocab"):
            raise ValueError(
                "Model must have src_vocab and tgt_vocab attached before inference."
            )
    
        src_vocab = self.src_vocab
        tgt_vocab = self.tgt_vocab
    
        # Fast tokenization
        #tokens = src_sentence.lower().strip().split()
        
        from spacy.lang.de import German

        if not hasattr(self, "_tokenizer"):
            self._tokenizer = German().tokenizer
        
        tokens = [
            tok.text.lower()
            for tok in self._tokenizer(src_sentence)
        ]
        
        # Numericalize
        unk_idx = src_vocab.stoi.get("<unk>", 0)
    
        src_ids = (
            [src_vocab.stoi["<sos>"]]
            + [src_vocab.stoi.get(tok, unk_idx) for tok in tokens]
            + [src_vocab.stoi["<eos>"]]
        )
    
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
        with torch.no_grad():
    
            src_mask = make_src_mask(src_tensor).to(device)
    
            # Encode once
            memory = self.encode(src_tensor, src_mask)
    
            # Start token
            ys = torch.LongTensor(
                [[tgt_vocab.stoi["<sos>"]]]
            ).to(device)
    
            eos_idx = tgt_vocab.stoi["<eos>"]
    
            # VERY IMPORTANT
            # Keep this small for autograder speed
            max_len = 30
    
            for _ in range(max_len):
    
                tgt_mask = make_tgt_mask(ys).to(device)
    
                out = self.decode(
                    memory,
                    src_mask,
                    ys,
                    tgt_mask
                )
    
                next_word = out[:, -1].argmax(dim=-1).item()
    
                # Stop immediately at EOS
                if next_word == eos_idx:
                    break
    
                ys = torch.cat(
                    [
                        ys,
                        torch.LongTensor([[next_word]]).to(device)
                    ],
                    dim=1
                )
    
            # Convert ids to tokens
            translated_tokens = []
    
            for idx in ys.squeeze(0).tolist()[1:]:
    
                if idx == eos_idx:
                    break
    
                token = tgt_vocab.itos[idx]
    
                if token not in ["<pad>", "<sos>", "<eos>"]:
                    translated_tokens.append(token)
    
        return " ".join(translated_tokens)
"""A complete GPT in one file.

The model reads a sequence of token IDs and, for every position, outputs a
score for each token in the vocabulary: how likely that token is to come next.

    token IDs -> vectors -> [ talk (attention), think (MLP) ] x n_layer -> scores
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 64  # the longest sequence the model can read
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128


class Attention(nn.Module):
    """Talk: every position gathers information from itself and earlier positions."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.query = nn.Linear(cfg.d_model, cfg.d_model)  # what am I looking for?
        self.key = nn.Linear(cfg.d_model, cfg.d_model)  # what do I contain?
        self.value = nn.Linear(cfg.d_model, cfg.d_model)  # what do I hand over if chosen?
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        # Position i may only look at positions <= i.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, trace=None, ablate_heads=()):
        B, T, D = x.shape
        # Split each D-wide vector into n_head smaller ones: (B, n_head, T, d_head).
        split = lambda t: t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))

        # Dot product of every query with every key, scaled to keep softmax well-behaved.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)  # each row sums to 1

        z = weights @ v  # weighted average of the values
        for h in ablate_heads:
            z[:, h] = 0.0
        y = self.out(z.transpose(1, 2).reshape(B, T, D))  # glue the heads back together

        if trace is not None:
            trace.update(q=q, k=k, v=v, scores=scores, attn=weights, z=z, attn_out=y)
        return y


class MLP(nn.Module):
    """Think: every position processes what it has gathered, on its own."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x, trace=None):
        pre = self.fc(x)
        act = F.gelu(pre, approximate="tanh")
        y = self.proj(act)
        if trace is not None:
            trace.update(mlp_pre=pre, mlp_act=act, mlp_out=y)
        return y


class Block(nn.Module):
    """One layer. Both steps add their result to the stream; neither replaces it."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, trace=None, ablate_heads=(), ablate_mlp=False):
        if trace is not None:
            trace["resid_pre"] = x
        x = x + self.attn(self.ln1(x), trace, ablate_heads)
        if trace is not None:
            trace["resid_mid"] = x
        if not ablate_mlp:
            x = x + self.mlp(self.ln2(x), trace)
        if trace is not None:
            trace["resid_post"] = x
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)  # one vector per token
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)  # one vector per position
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)
        # Scale down the layers that write into the stream, as GPT-2 does.
        for name, p in self.named_parameters():
            if name.endswith(("attn.out.weight", "mlp.proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, trace=None, no_pos=False, ablate_heads=(), ablate_mlp=()):
        """idx: (B, T) token IDs. Returns scores (B, T, vocab_size) and the loss.

        ablate_heads is a collection of (layer, head) pairs; ablate_mlp a collection of layers.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.wte(idx)
        if not no_pos:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        if trace is not None:
            trace["tok_emb"] = self.wte(idx)
            trace["embed"] = x
            trace["layers"] = []

        for i, block in enumerate(self.blocks):
            layer_trace = {} if trace is not None else None
            heads = [h for (l, h) in ablate_heads if l == i]
            x = block(x, layer_trace, heads, i in ablate_mlp)
            if trace is not None:
                trace["layers"].append(layer_trace)

        x = self.ln_f(x)
        # The output matrix is the embedding table again ("weight tying"):
        # a token scores highly when the final vector points the same way as its embedding.
        logits = x @ self.wte.weight.T
        if trace is not None:
            trace.update(final=x, logits=logits)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1] / temperature  # only the last position predicts
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits[logits < kth] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

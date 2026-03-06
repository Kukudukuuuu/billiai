"""
BillaiAI (Small Language Model) from scratch
============================================
A minimal GPT-style transformer you can train on your own text.

Requirements:
    pip install torch

Usage:
    # Train on your own text file:
    python billiai.py --train --file yourtext.txt

    # Or train on the built-in sample text:
    python billiai.py --train

    # Generate text after training:
    python billiai.py --generate --prompt "Once upon a time"
"""

import os
import math
import argparse
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F

# ─────────────────────────────────────────────
#  CONFIG  (tweak these to make model bigger/smaller)
# ─────────────────────────────────────────────
CONFIG = {
    # Model architecture
    "vocab_size":    None,   # set automatically from data
    "context_len":   128,    # how many tokens the model sees at once
    "n_embed":       128,    # embedding dimension
    "n_heads":       4,      # number of attention heads
    "n_layers":      4,      # number of transformer blocks
    "dropout":       0.1,

    # Training
    "batch_size":    32,
    "max_steps":     3000,
    "eval_interval": 300,
    "eval_steps":    50,
    "lr":            3e-4,

    # Files
    "data_file":     "input.txt",
    "model_file":    "billiai.pt",
    "tokenizer_file":"billiai_tokenizer.pkl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
#  SAMPLE TRAINING TEXT (used if no file given)
# ─────────────────────────────────────────────
SAMPLE_TEXT = """
Human: hi
Billi AI: hello! how are you?
Human: i am good thanks
Billi AI: that is great to hear!
Human: what is your name?
Billi AI: my name is Billi AI. nice to meet you!
Human: who made you?
Billi AI: i was made from scratch by a developer.
Human: how are you?
Billi AI: i am doing well thank you for asking!
Human: what can you do?
Billi AI: i can chat with you and answer basic questions.
Human: tell me a joke
Billi AI: why did the computer go to the doctor? because it had a virus!
Human: that is funny
Billi AI: thank you! i am glad you liked it.
Human: what is the weather like?
Billi AI: i do not know the weather but i hope it is nice where you are!
Human: do you like music?
Billi AI: i think music is wonderful. what kind do you like?
Human: i like pop music
Billi AI: pop music is very popular. do you have a favourite artist?
Human: yes i like taylor swift
Billi AI: she is very talented and has many great songs.
Human: what is your favourite colour?
Billi AI: i like blue. it is a calm and peaceful colour.
Human: mine is red
Billi AI: red is a bold and energetic colour. great choice!
Human: are you smart?
Billi AI: i try my best to be helpful and learn from our conversations.
Human: can you help me?
Billi AI: of course! i am always happy to help. what do you need?
Human: i need some advice
Billi AI: sure, tell me what is on your mind and i will do my best.
Human: thank you
Billi AI: you are welcome! i am here whenever you need me.
Human: goodbye
Billi AI: goodbye! it was nice talking to you. have a great day!
Human: see you later
Billi AI: see you later! take care!
Human: hi again
Billi AI: welcome back! it is good to see you again.
Human: i missed you
Billi AI: that is very kind! i missed chatting with you too.
Human: what time is it?
Billi AI: i do not have a clock but you can check your phone for the time.
Human: good idea
Billi AI: glad i could help! is there anything else you want to talk about?
Human: not really
Billi AI: that is okay. just let me know whenever you want to chat!
Human: you are cool
Billi AI: thank you so much! you are cool too.
Human: can you learn?
Billi AI: yes i learn from the conversations i am trained on.
Human: that is interesting
Billi AI: i think so too! language and learning are fascinating things.
Human: do you have feelings?
Billi AI: i do not have real feelings but i try to be kind and friendly.
Human: that is okay
Billi AI: thank you for understanding! i appreciate your kindness.
Human: what do you think about AI?
Billi AI: i think AI is a powerful tool that can help people in many ways.
Human: do you think AI is dangerous?
Billi AI: it can be if used badly but mostly it is meant to help people.
Human: i agree
Billi AI: great! it is important to think about these things carefully.
Human: you are helpful
Billi AI: thank you! being helpful is my favourite thing to do.
Human: hi
Billi AI: hey there! what is up?
Human: nothing much
Billi AI: that sounds relaxing! sometimes doing nothing is the best thing.
Human: true
Billi AI: how has your day been so far?
Human: pretty good
Billi AI: happy to hear that! i hope it keeps going well.
Human: what is your favourite food?
Billi AI: i do not eat food but if i could i think i would enjoy pizza!
Human: pizza is great
Billi AI: it really is! so many toppings to choose from.
Human: i like cheese pizza
Billi AI: classic choice! you cannot go wrong with cheese pizza.
Human: do you like games?
Billi AI: i think games are a fun way to spend time and challenge your mind.
Human: i like video games
Billi AI: video games are very creative. what kind do you like to play?
Human: i like adventure games
Billi AI: adventure games are exciting! exploring new worlds is so fun.
Human: yes exactly
Billi AI: sounds like you have great taste in games!
Human: thanks billi
Billi AI: anytime! always happy to chat with you.
"""


# ─────────────────────────────────────────────
#  TOKENIZER  (character-level, simple & fast)
# ─────────────────────────────────────────────
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(i, "?") for i in ids)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
#  MODEL COMPONENTS
# ─────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, n_embed, n_heads, context_len, dropout):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads

        self.qkv  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)

        # causal mask: tokens only attend to past tokens
        mask = torch.tril(torch.ones(context_len, context_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # reshape to (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Simple 2-layer MLP with GELU activation."""
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block = attention + feedforward, with residuals."""
    def __init__(self, n_embed, n_heads, context_len, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embed)
        self.attn = SelfAttention(n_embed, n_heads, context_len, dropout)
        self.ln2  = nn.LayerNorm(n_embed)
        self.ff   = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection
        x = x + self.ff(self.ln2(x))     # residual connection
        return x


class BilliAI(nn.Module):
    """The full Billi AI language model."""
    def __init__(self, cfg):
        super().__init__()
        V = cfg["vocab_size"]
        C = cfg["n_embed"]
        T = cfg["context_len"]

        self.tok_embed = nn.Embedding(V, C)
        self.pos_embed = nn.Embedding(T, C)
        self.drop      = nn.Dropout(cfg["dropout"])
        self.blocks    = nn.Sequential(*[
            TransformerBlock(C, cfg["n_heads"], T, cfg["dropout"])
            for _ in range(cfg["n_layers"])
        ])
        self.ln_f  = nn.LayerNorm(C)
        self.head  = nn.Linear(C, V, bias=False)

        # weight tying: share token embedding & output weights
        self.head.weight = self.tok_embed.weight

        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        print(f"Billi AI ready — {total:,} parameters ({total/1e6:.2f}M)")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)

        x = self.drop(self.tok_embed(idx) + self.pos_embed(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Autoregressively generate tokens."""
        context_len = self.pos_embed.num_embeddings
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────

def get_batch(data, cfg):
    """Sample a random batch of (input, target) token sequences."""
    T  = cfg["context_len"]
    B  = cfg["batch_size"]
    ix = torch.randint(len(data) - T, (B,))
    x  = torch.stack([data[i:i+T]   for i in ix])
    y  = torch.stack([data[i+1:i+T+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    model.eval()
    results = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg["eval_steps"]):
            x, y = get_batch(data, cfg)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

def train(cfg):
    # Load or create text
    if os.path.exists(cfg["data_file"]):
        with open(cfg["data_file"], "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded '{cfg['data_file']}' — {len(text):,} characters")
    else:
        print("No data file found — using built-in sample text.")
        text = SAMPLE_TEXT

    # Tokenize
    tokenizer = CharTokenizer(text)
    tokenizer.save(cfg["tokenizer_file"])
    cfg["vocab_size"] = tokenizer.vocab_size
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # Build model
    model = BilliAI(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    print(f"\nBilli AI training on {DEVICE} for {cfg['max_steps']} steps...\n")

    for step in range(cfg["max_steps"] + 1):
        # Periodic evaluation
        if step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg)
            print(f"Step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        if step == cfg["max_steps"]:
            break

        x, y = get_batch(train_data, cfg)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Save
    torch.save({"model_state": model.state_dict(), "config": cfg}, cfg["model_file"])
    print(f"\nBilli AI saved to '{cfg['model_file']}'")

    # Quick sample
    print("\n── Sample generation ──")
    generate(cfg, prompt="\n", max_tokens=200, model=model, tokenizer=tokenizer)


# ─────────────────────────────────────────────
#  GENERATION
# ─────────────────────────────────────────────

def generate(cfg, prompt="", max_tokens=300, temperature=0.8, top_k=40,
             model=None, tokenizer=None):
    # Load if not passed in
    if model is None:
        checkpoint = torch.load(cfg["model_file"], map_location=DEVICE)
        saved_cfg  = checkpoint["config"]
        tokenizer  = CharTokenizer.load(saved_cfg["tokenizer_file"])
        saved_cfg["vocab_size"] = tokenizer.vocab_size
        model = BilliAI(saved_cfg).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])

    model.eval()
    context = torch.tensor(tokenizer.encode(prompt or "\n"),
                           dtype=torch.long, device=DEVICE).unsqueeze(0)

    out = model.generate(context, max_tokens, temperature=temperature, top_k=top_k)
    decoded = tokenizer.decode(out[0].tolist())
    # Only show the first response line, not the whole conversation
    if "Human:" in decoded[len(prompt):]:
        reply = decoded[len(prompt):decoded.index("Human:", len(prompt))]
    else:
        reply = decoded[len(prompt):]
    print(reply.strip())


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Billi AI from scratch")
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--file",     type=str, help="Path to training text file")
    parser.add_argument("--prompt",   type=str, default="", help="Prompt for generation")
    parser.add_argument("--tokens",   type=int, default=300, help="Tokens to generate")
    parser.add_argument("--temp",     type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    if args.file:
        CONFIG["data_file"] = args.file

    if args.train:
        train(CONFIG)
    elif args.generate:
        generate(CONFIG, prompt=args.prompt, max_tokens=args.tokens, temperature=args.temp)
    else:
        print(__doc__)

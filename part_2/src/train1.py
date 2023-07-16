import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math


class char_tokenizer:
    """
    A very simple char-based tokenizer. The tokenizer turns a string into a list of integers.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.n_vocab = len(corpus)
        self.char_to_int = {char: i for i, char in enumerate(corpus)}

    def encode(self, string: str):
        return [self.char_to_int[char] for char in string]

    def decode(self, codes: List[int]):
        return ''.join([self.corpus[code] for code in codes])


class Head(nn.Module):
    """Single head of self-attention"""

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.Key = nn.Linear(n_embd, head_size)
        self.Query = nn.Linear(n_embd, head_size)
        self.Value = nn.Linear(n_embd, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs):
        key = self.Key(inputs)
        query = self.Query(inputs)
        value = self.Value(inputs)

        # Compute attention scores
        out = torch.matmul(query, key.transpose(-2, -1))
        out = out / (key.size(-1) ** 0.5)  # Scale by square root of key size
        # out = out * self.tril[: out.size(-1) , :out.size(-1)]
        out = out.masked_fill(self.tril[:out.size(-1), :out.size(-1)] == 0, float('-inf'))
        out = F.softmax(out, dim=-1)
        out = torch.matmul(out, value)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embd, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, n_embd)

    def forward(self, inputs):
        head_outputs = [head(inputs) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        out = self.projection(concatenated)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, inputs):
        return self.net(inputs)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, n_embd, n_embd // n_heads)
        self.attention_norm = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd)
        self.feed_forward_norm = nn.LayerNorm(n_embd)

    def forward(self, inputs):
        attn_output = self.attention(inputs)
        attn_output = attn_output + inputs
        attn_output = self.attention_norm(attn_output)

        ff_output = self.feed_forward(attn_output)
        ff_output = ff_output + attn_output
        ff_output = self.feed_forward_norm(ff_output)

        return ff_output


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_heads) for _ in range(n_layers)])
        self.position_encoding = self._create_positional_encoding(n_embd, block_size)
        #self.norm = nn.LayerNorm(n_embd)
        self.linear=nn.Linear(n_embd,n_vocab)

    def _create_positional_encoding(self, n_embd, max_sequence_length):
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        encoding = torch.zeros((max_sequence_length, n_embd))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding
    
    def forward(self, inputs, labels=None):
        embedding = self.embedding(inputs)
        embedding += self.position_encoding[:embedding.size(1),:].to(inputs.device)

        attens = self.blocks[0](embedding)
        for block in self.blocks[1:]:
            attens = block(attens)

        #mid = self.linear_in(attens)
        #logits = attens + self.linear_out(mid)
        #logits = self.norm(logits)
        logits=self.linear(attens)

        if labels is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)
            labels = labels.view(batch * time)
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            output, _ = self(inputs)
            next_token = torch.argmax(output[:, -1, :], dim=-1)
            inputs = torch.cat((inputs, next_token[:, None]), dim=1)
        return inputs


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def generate(model):
    context = torch.tensor(encode('To be or not to be'), device=device, dtype=torch.long).unsqueeze(0)
    str = tokenizer.decode(model.generate(context, max_new_tokens=200)[0].tolist())
    print(str)
    print(len(str))


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        inputs, labels = get_batch("train")

        logits, loss = model(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


# Define the hyperparameters
batch_size = 16
block_size = 256
max_iters = 5000  # Set the number of training iterations as you like
#max_iters=1
eval_interval = 50
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
eval_iters = 200
n_embd = 64
n_heads = 8
n_layers = 6

# Read the dataset
with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))

# Initialize the vocabulary
tokenizer = char_tokenizer(chars)
encode = tokenizer.encode
decode = tokenizer.decode
n_vocab = tokenizer.n_vocab

# Separate the dataset into train and validation
train_data = torch.tensor(encode(text[: -len(text) // 10]), dtype=torch.long)
val_data = torch.tensor(encode(text[-len(text) // 10:]), dtype=torch.long)

# Define the model
model = Transformer().to(device)
train(model)
generate(model)
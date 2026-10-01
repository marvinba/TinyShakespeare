import torch
import torch.nn as nn
import torch.nn.functional as F
from config import block_size, vocab_size, dialogue_id

class Head(nn.Module):

    def __init__(self, dropout, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1) * self.head_size**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        v = self.value(x)
        out = weights @ v
        return out

class TransformerModel(nn.Module):

    def __init__(self, dropout, n_embd, n_head, n_layer, label_smoothing):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(dropout, n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.label_smoothing = label_smoothing

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape

        tok_emb = self.token_embedding_table(token_ids)
        pos_emb = self.pos_embedding_table(torch.arange(T,device=token_ids.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets, label_smoothing=self.label_smoothing if self.training else 0.0)

        return logits,loss

    @torch.no_grad()
    def generate(self, id):
        
        #disable dropout during generation
        self.eval()

        while True:
            logits, _ = self.forward(id)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == dialogue_id and id.size(1) == 1:
                continue

            id = torch.cat([id, next_id], dim=1)

            if id.size(1) == block_size:
                break

            if next_id.item() == dialogue_id:
                break
        return id

class MultiHeadAttention(nn.Module):

    def __init__(self, dropout, n_embd, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(dropout, n_embd, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, dropout, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(dropout, n_embd, n_head, head_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

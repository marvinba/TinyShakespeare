import torch
import numpy
import re as regExp
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

#load tokens from dataset
dialogueStr = '<Dialogue>'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = '\n\n' + f.read();
    text = regExp.sub(r'[\n]{2}[\w ]+[:]{1}', dialogueStr, text)
    
#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 320
n_head = 8
n_layer = 4
dropout = 0.30
learning_rate = 3e-4
vocab_size = 1000
dialogue_id = 256
weight_decay = 0.0
warmup_iter = 250

#split dataset
split_idx = int(0.9 * len(text))
split_idx = text.find("<Dialogue>", split_idx)
train_text = text[:split_idx]
val_text = text[split_idx:]

class BPETokenizer():

    def __init__(self, data):
        self.token_ids = list(data.encode("utf-8")) #list of ids of each byte        
        self.merges = {}
        self.idToToken = {}
        self.num_merges = vocab_size - 257
        self.replace_dialogueID(self.token_ids)
        self.bpe_merge()
        self.idsToTokens()
        self.tokenToId = {
            token: token_id
            for token_id, token in self.idToToken.items()
        }        
      
    def replace_dialogueID(self, byteIDs):
        dialogueIDs = list(dialogueStr.encode("utf-8"))

        i = 0
        dialogueSz = len(dialogueIDs)
        while i <= len(byteIDs) - dialogueSz:
            if (byteIDs[i:i+dialogueSz] == dialogueIDs): 
                byteIDs[i:i+dialogueSz] = [dialogue_id]
            i+=1

    def find_pair(self):
        counts = {}
        for pair in zip(self.token_ids, self.token_ids[1:]):
            if dialogue_id in pair:
                continue

            counts[pair] = counts.get(pair, 0) + 1

        most_frequent_pair = max(counts, key=counts.get)

        return most_frequent_pair

    def bpe_merge(self):

        for i in range(1, self.num_merges+1):
            pair = self.find_pair()
            tokenid = 256 + i
            self.merges[tokenid] = pair

            j = 0
            while(j < len(self.token_ids)- 1):

                if((self.token_ids[j] == pair[0]) and (self.token_ids[j+1] == pair[1])):
                    self.token_ids[j:j+2] = [tokenid]
                else:
                    j+=1
        
    def idsToTokens(self):

        for i in range(256):
            self.idToToken[i] = bytes([i]) #id 0-255 is its bytes 

        self.idToToken[dialogue_id] = b'<Dialogue>'

        for i in range (257, vocab_size): #257 or greater in merges
            pair = self.merges[i] # gives you a pair
            self.idToToken[i] = self.idToToken[pair[0]] + self.idToToken[pair[1]]

    def encode(self, text):
        token_ids = list(text.encode("utf-8"))
        self.replace_dialogueID(token_ids)

        for token_id, pair in self.merges.items():

            idx = 0
            while idx < len(token_ids) - 1:
                if pair == (token_ids[idx], token_ids[idx+1]):
                    token_ids[idx:idx+2] = [token_id]
                else:
                    idx+=1

        return torch.tensor(token_ids, dtype=torch.long)
    
    def getToken(self, token_id):
        return self.idToToken[token_id]

tokenizer = BPETokenizer(train_text)
train_data = tokenizer.encode(train_text)
val_data = tokenizer.encode(val_text)

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

    def __init__(self, dropout, n_embd, n_head, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(dropout, n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape

        tok_emb = self.token_embedding_table(token_ids)
        pos_emb = self.pos_embedding_table(torch.arange(T,device=device))

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
            loss = F.cross_entropy(logits,targets)

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

def get_batch(encoded_data, batch_size):
    batch_ids = []
    target_ids = []

    #picks batch_size starting indicies, leaves room for shifted target
    starts = torch.randint(0, len(encoded_data) - block_size, (batch_size,))

    #generate batch_size sequences of block_size tokens
    for i in range (batch_size):

        #Get input sequence of block_size tokens
        seq = encoded_data[starts[i]:starts[i] + block_size]

        #Target sequence shifted one token forward
        target = encoded_data[starts[i] + 1: starts[i] + block_size + 1]

        batch_ids.append(seq)
        target_ids.append(target)

    #Combine individual sequences into one tensor
    batch_ids = torch.stack(batch_ids).to(device)
    target_ids = torch.stack(target_ids).to(device)

    return batch_ids, target_ids

def estimate_loss(model):
    
    losses = {}

    #Put model into evaluation mode
    model.eval()

    #Don't calculate gradients during evaluation
    with torch.no_grad():
        
        for split, data in [('train', train_data), ('val', val_data)]:

            split_losses = []

            for _ in range(eval_iters):

                batch_ids, target_ids = get_batch(data, batch_size)

                #Forward pass
                _, loss = model(batch_ids, target_ids)

                split_losses.append(loss.item())

            losses[split] = torch.tensor(split_losses).mean()

    # Put model back into training mode
    model.train()

    return losses

min_validation_loss = float('inf')

model = TransformerModel(dropout, n_embd, n_head, n_layer).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

if warmup_iter == 0:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_iters
    )

else:
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_iter
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_iters - warmup_iter
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iter]
    )

for iter in range(max_iters+1):  

    if iter % eval_interval == 0:
        losses = estimate_loss(model)

        print(
            f"Iteration {iter}, "
            f"Training loss = {losses['train']:.4f}, "
            f"Validation loss = {losses['val']:.4f}"
        )

        if losses['val'] < min_validation_loss:
            min_validation_loss = losses['val']
            training_loss = losses['train']

            best_config = {
                'iteration': iter
            }

    if iter == max_iters:
        break
                
    batch_ids, target_ids = get_batch(train_data, batch_size)

    #Forward pass
    logits, loss = model(batch_ids, target_ids)

    #Backpropagation
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    #Update weights and decay learning rate
    optimizer.step()
    scheduler.step()
            
genIds = model.generate(torch.tensor([[dialogue_id]],dtype=torch.long,device=device))
genTokens = [
    tokenizer.getToken(token_id)
    for token_id in genIds[0].tolist()
    if token_id != dialogue_id
]

generated_bytes = b''.join(genTokens)
generated_text = generated_bytes.decode('utf-8', errors='replace')
generated_text = generated_text + '\n'

print(generated_text)

print("Configuration with lowest validation:\n"
    f"Iteration {best_config['iteration']}, "
    f"Validation loss = {min_validation_loss:.4f}, "
    f"Training loss = {training_loss:.4f}"
)
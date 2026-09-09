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
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192
n_head = 6
n_layer = 4
dropout = 0.2

#tokenizer hyperparameters
vocab_size = 1000
num_merges = vocab_size - 257

token_ids = list(text.encode("utf-8")) #list of raw bytes
dialogueBytes = list(dialogueStr.encode("utf-8"))
dialogue_id = 256
merges = {}
idToToken = {}

def replace_dialogue():
    i = 0
    dialogueSz = len(dialogueBytes)
    while i <= len(token_ids) - dialogueSz:
        if (token_ids[i:i+dialogueSz] == dialogueBytes): 
            token_ids[i:i+dialogueSz] = [dialogue_id]
        i+=1

replace_dialogue()

def find_pair(data):
    counts = {}
    for pair in zip(data, data[1:]):
        if dialogue_id in pair:
            continue

        counts[pair] = counts.get(pair, 0) + 1
    most_frequent_pair = max(counts, key=counts.get)

    return most_frequent_pair

def bpe_merge():

    for i in range(1, num_merges+1):
        pair = find_pair(token_ids)
        new_token = 256 + i
        merges[new_token] = pair

        j = 0
        while(j < len(token_ids)- 1):

            if((token_ids[j] == pair[0]) and (token_ids[j+1] == pair[1])):
                token_ids[j:j+2] = [new_token]
            j+=1
    
bpe_merge()

def convertToToken(id):
    if id > 256:
        return merges[id]
    else:
        return idToToken[id]
    
def idsToTokens():

    for i in range(256):
        idToToken[i] = chr(i) #0-255 is its char

    idToToken[dialogue_id] = '<Dialogue>'

    for i in range (257, vocab_size): #257 or greater in merges
        pair = merges[i] # gives you a pair
        idToToken[i] = convertToToken(pair[0]) + convertToToken(pair[1])

idsToTokens()

#split dataset
n = int(0.9 * len(token_ids))
train_data = token_ids[:n]
val_data = token_ids[n:]

class Head(nn.Module):

    def __init__(self,head_size):
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

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
    
    def generate(self, idx):
        
        #disable dropout during generation
        self.eval()

        while True:
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if idx.size(1) == block_size:
                break

            if next_token.item() == dialogue_id:
                break
        return idx


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


ShakespearenModel =  TransformerModel().to(device)  

optimizer = torch.optim.AdamW(
    ShakespearenModel.parameters(),
    lr=learning_rate
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_iters
)

def get_batch(data):
    batch_ids = []
    targets = []


    #picks batch_size starting indicies, leaves room for shifted target
    starts = torch.randint(0, len(data) - block_size, (batch_size,))

    #generate batch_size sequences of block_size tokens
    for i in range (batch_size):

        #Get input sequence of block_size tokens
        sequence = data[starts[i]:starts[i] + block_size]

        #Target sequence shifted one token forward
        target = data[starts[i] + 1: starts[i] + block_size + 1]

        batch_ids.append(sequence)
        targets.append(target)

    #Combine individual sequences into one tensor
    batch_ids = torch.stack(batch_ids).to(device)
    targets = torch.stack(targets).to(device)

    return batch_ids, targets

def estimate_loss():
    
    losses = {}

    #Put model into evaluation mode
    ShakespearenModel.eval()

    #Don't calculate gradients during evaluation
    with torch.no_grad():
        
        for split, data in [('train', train_data), ('val', val_data)]:

            split_losses = []

            for _ in range(eval_iters):

                batch_ids, targets = get_batch(data)

                #Forward pass
                _, loss = ShakespearenModel(batch_ids, targets)

                split_losses.append(loss.item())

            losses[split] = torch.tensor(split_losses).mean()

    # Put model back into training mode
    ShakespearenModel.train()

    return losses

for iter in range(max_iters+1):    

    if iter % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Iteration {iter}: "
            f"Training loss = {losses['train']:.4f}, "
            f"Validation loss = {losses['val']:.4f}"
        )

    batch_ids, targets = get_batch(train_data)

    #Forward pass
    logits, loss = ShakespearenModel(batch_ids, targets)

    #Backpropagation
    optimizer.zero_grad()
    loss.backward()

    #Update weights and decay learning rate
    optimizer.step()
    scheduler.step()

#generate text after training (go from <dialogue> to <dialogue>)
genIds = ShakespearenModel.generate(torch.tensor([[dialogue_id]],dtype=torch.long,device=device))
genTokens = [
    idToToken[token_id]
    for token_id in genIds[0].tolist()
    if idToToken[token_id] != '<Dialogue>'
]

generated_text = ' '.join(genTokens)
generated_text = regExp.sub(r'\s+([.?!,:;])', r'\1', generated_text)

print(generated_text)


import torch
import numpy
import re as regExp
import random
import torch.nn.functional as F

#load tokens from dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = '\n\n' + f.read();
    text = regExp.sub(r'[\n]{2}[\w ]+[:]{1}', '<Dialogue>', text[:50000])
    tokenArr = regExp.findall(r'[<]{1}[\w]+[>]{1}|[.?!,:;]|[\w]+', text) #create tokens including <Dialogue>
    distTokenDict = {token:idx for idx, token in enumerate(dict.fromkeys(tokenArr))}

#load sequences from data in terms of indices
seq_length = 5 #5
X, Y = [], []
for token in tokenArr:
    if token == '<Dialogue>':
        context = [0] * seq_length
        continue
    idx = distTokenDict[token]
    X.append(context)
    Y.append(idx)
    context = context[1:] + [idx]

#print(distTokenDict)
X = torch.tensor(X) #(237803, 5)
Y = torch.tensor(Y) #(237803)
    
#create lookup table, hidden nonlinearity layer, the last layer being linear, and biases for each layer. All parameters which we'll use Use generator for 
g = torch.Generator().manual_seed(5000)
embed_dim = 8 #25
num_tokens = len(distTokenDict)
num_neurons = 25
lookupTbl = torch.randn((num_tokens, embed_dim),generator=g) #weight matrix where you index to each token that has an embedding of 25 dimensions. There are 13113 distinct tokens
W1 = torch.randn((seq_length * embed_dim, num_neurons), generator=g) #100 neurons
b1 = torch.randn(num_neurons, generator=g)
W2 = torch.randn((num_neurons, num_tokens), generator=g) #outputs number of neurons is num_tokens since we have num_tokens possible tokens that come next 
b2 = torch.randn(num_tokens, generator=g)
parameters = [lookupTbl, W1, b1, W2, b2] #1664838 total parameters

for p in parameters:
    p.requires_grad = True

for i in range(1000):
    #minibatch construct
    ix = torch.randint(0, X.shape[0], (32,)) #minibatch size of 32

    #forward pass
    emb = lookupTbl[X[ix]] # (237803, 5, 25)
    h = torch.tanh(emb.view(-1, seq_length * embed_dim) @ W1 + b1) #(237803, 100)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    
    print(loss.item())
    
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    

#batch
#sequence length - context length of how many words we take to predict the next one


#how does sequence length and batches work in tandem?
#"a lot of input examples in a batch?"

#"to randomly select some portion of the data set and 
# that’s a mini batch and then only forward backward 
# and update on that little mini batch and then we 
# iterate on those many batches."

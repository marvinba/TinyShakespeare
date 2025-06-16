import torch
import numpy
import re as regExp
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

#load tokens from dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = '\n\n' + f.read();
    text = regExp.sub(r'[\n]{2}[\w ]+[:]{1}', '<Dialogue>', text[:45000])
    tokenArr = regExp.findall(r'[<]{1}[\w]+[>]{1}|[.?!,:;]|[\w]+', text) #create tokens including <Dialogue>
    distTokenDict = {token:idx for idx, token in enumerate(dict.fromkeys(tokenArr))}

#print("Equal probability for any token " + str(-torch.tensor(1/len(distTokenDict)).log().item()))

seq_length =  7
def build_dataset(tokens):
    #load sequences from data in terms of indices
    X, Y = [], []
    dialogueIdx = distTokenDict['<Dialogue>']
    context = [dialogueIdx] * seq_length

    for token in tokens:
        if token == '<Dialogue>':
            context = [dialogueIdx] * seq_length
            continue
        
        idx = distTokenDict[token]
        X.append(context)
        Y.append(idx)
        context = context[1:] + [idx]
        
    X = torch.tensor(X) 
    Y = torch.tensor(Y) 
    return X,Y

#Training, Dev, and Testing splits
trEndIdx = int(0.8*len(tokenArr))
devEndIdx = int(0.9*len(tokenArr))
Xtr, Ytr = build_dataset(tokenArr[:trEndIdx])
Xdev, Ydev = build_dataset(tokenArr[trEndIdx:devEndIdx])
Xtest, Ytest = build_dataset(tokenArr[devEndIdx:])

#create lookup table, hidden nonlinearity layer, the last layer being linear, and biases for each layer. All parameters which we'll use Use generator for 
g = torch.Generator().manual_seed(5000)
embed_dim = 7
num_tokens = len(distTokenDict)
num_neurons1 = 10
num_neurons2 = 11
num_neurons3 = 9
lookupTbl = torch.randn((num_tokens, embed_dim),generator=g) #weight matrix where you index to each token that has an embedding of 25 dimensions. There are 13113 distinct tokens
W1 = torch.randn((seq_length * embed_dim, num_neurons1), generator=g) * 0.1
b1 = torch.randn(num_neurons1, generator=g) * 0.01
W2 = torch.randn((num_neurons1, num_tokens), generator=g) * 0.06 #torch.randn((num_neurons1, num_neurons2), generator=g) #outputs number of neurons is num_tokens since we have num_tokens possible tokens that come next 
b2 = torch.randn(num_tokens, generator=g) * 0.1 #torch.randn(num_neurons2, generator=g)

bngain = torch.ones((1, num_neurons1))
bnbias = torch.zeros((1, num_neurons1))
bnmean_running = torch.zeros((1, num_neurons1)) #mean is 0 for unit gaussian
bnstd_running = torch.zeros((1, num_neurons1))  #std is 1 for unit gaussian
parameters = [lookupTbl, W1, b1, W2, b2, bngain, bnbias]#, W3, b3] #76199 total parameters  W4, b4


lri = []
lossi = []
dlossi = [] 
steps = []

for p in parameters:
    p.requires_grad = True

num_iterations = 100000
for i in range(num_iterations):
    #minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (200,)) #updated minibatch size 

    #forward pass - batch training loss
    emb = lookupTbl[Xtr[ix]]
    hpreact = emb.view(-1, seq_length * embed_dim) @ W1 + b1
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True) 
    hpreact = bngain * (hpreact - bnmeani)/bnstdi + bnbias #batchnorm layer
    
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 *bnmeani
        bnstd_running = 0

    h = torch.tanh(hpreact) #(237803, 100)
    logits = h @ W2 + b2 #W3 + b3 # @ W4 + b4
    loss = F.cross_entropy(logits, Ytr[ix])

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    #lr = lrs[i]
    lr = 0.05

    for p in parameters:
        p.data += -lr * p.grad
 
    #lri.append([lre[i]])
    lossi.append(loss.item())
    #dlossi.append(lossDev.item())
    steps.append(i)

#batch training loss
print("batch training loss " + str(loss.item()))

@torch.no_grad()
def split_loss(split):
    x,y = { 
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xtest, Ytest),
    }[split]
    emb = lookupTbl[x]
    hpreact = emb.view(-1, seq_length * embed_dim) @ W1 + b1
    #hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True))/hpreact.std(0, keepdim=True) + bnbias
    hpreact = bngain * (hpreact - bnmean_running)/bnstd_running + bnbias

    h = torch.tanh((emb.view(-1, seq_length * embed_dim) @ W1 + b1)) #@ W2 + b2)) #torch.tanh(((emb.view(-1, seq_length * embed_dim) @ W1 + b1) @ W2 + b2) @ W3 + b3) #(237803, 100)
    logits = h @ W2 + b2 #W3 + b3 #h @ W4 + b4
    loss = F.cross_entropy(logits, y)
    print(split, (loss.item()))

split_loss('train')
split_loss('val')



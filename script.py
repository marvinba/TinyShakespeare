import torch
import numpy
import re as regExp
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

#load tokens from dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = '\n\n' + f.read();
    text = regExp.sub(r'[\n]{2}[\w ]+[:]{1}', '<Dialogue>', text[:50000])
    tokenArr = regExp.findall(r'[<]{1}[\w]+[>]{1}|[.?!,:;]|[\w]+', text) #create tokens including <Dialogue>
    distTokenDict = {token:idx for idx, token in enumerate(dict.fromkeys(tokenArr))}

seq_length = 5
def build_dataset(tokens):
    #load sequences from data in terms of indices
    X, Y = [], []
    context = [0] * seq_length

    for token in tokens:
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
    return X,Y

#Training, Dev, and Testing splits
trEndIdx = int(0.8*len(tokenArr))
devEndIdx = int(0.9*len(tokenArr))
Xtr, Ytr = build_dataset(tokenArr[:trEndIdx])
Xdev, Ydev = build_dataset(tokenArr[trEndIdx:devEndIdx])
Xtest, Ytest = build_dataset(tokenArr[devEndIdx:])

#create lookup table, hidden nonlinearity layer, the last layer being linear, and biases for each layer. All parameters which we'll use Use generator for 
g = torch.Generator().manual_seed(5000)
embed_dim = 8 
num_tokens = len(distTokenDict)
num_neurons = 75
lookupTbl = torch.randn((num_tokens, embed_dim),generator=g) #weight matrix where you index to each token that has an embedding of 25 dimensions. There are 13113 distinct tokens
W1 = torch.randn((seq_length * embed_dim, num_neurons), generator=g) #100 neurons
b1 = torch.randn(num_neurons, generator=g)
W2 = torch.randn((num_neurons, num_tokens), generator=g) #outputs number of neurons is num_tokens since we have num_tokens possible tokens that come next 
b2 = torch.randn(num_tokens, generator=g)
parameters = [lookupTbl, W1, b1, W2, b2] #76199 total parameters  print(sum(p.nelement() for p in parameters))


#lre = torch.linspace(0.5,0.8,10000)
#lrs = 10**lre
#lri = []
#lossi = [] 

for p in parameters:
    p.requires_grad = True

for i in range(5000):
    #minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (500,)) #updated minibatch size 

    #forward pass
    emb = lookupTbl[Xtr[ix]] # (237803, 5, 25)
    h = torch.tanh(emb.view(-1, seq_length * embed_dim) @ W1 + b1) #(237803, 100)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    #lr = lrs[i]
    lr = 4
    for p in parameters:
        p.data += -lr * p.grad
        
    #if (i%1000 == 0 or i == 6999):
    #    print("iteration: " + str(i) + " loss: " + str(loss.item()))

    #track learning rate exponent and loss
    #lri.append([lre[i]])
    #lossi.append(loss.item())

#batch training loss
print("batch training loss " + str(loss.item()))

#training loss
emb = lookupTbl[Xtr] # (237803, 5, 25)
h = torch.tanh(emb.view(-1, seq_length * embed_dim) @ W1 + b1) #(237803, 100)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print("training set loss " + str(loss.item()))

#dev loss
emb = lookupTbl[Xdev] # (237803, 5, 25)
h = torch.tanh(emb.view(-1, seq_length * embed_dim) @ W1 + b1) #(237803, 100)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print("dev loss " + str(loss.item()))


#plot learning rate exponents vs losses to find right learning rate to use
#plt.plot(lri, lossi)
#plt.show()


#batch
#sequence length - context length of how many words we take to predict the next one


#how does sequence length and batches work in tandem?
#"a lot of input examples in a batch?"

#"to randomly select some portion of the data set and 
# that’s a mini batch and then only forward backward 
# and update on that little mini batch and then we 
# iterate on those many batches."

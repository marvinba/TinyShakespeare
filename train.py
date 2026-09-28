import torch
import numpy
import re as regExp
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from config import dialogueStr, dialogue_id, device, block_size, dropout, n_embd, n_head, n_layer, label_smoothing, learning_rate, weight_decay, max_iters, eval_interval, eval_iters, batch_size, warmup_iter
from tokenizer import BPETokenizer
from model import TransformerModel

#load tokens from dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = '\n\n' + f.read();
    text = regExp.sub(r'[\n]{2}[\w ]+[:]{1}', dialogueStr, text)
    
#split dataset
split_idx = int(0.9 * len(text))
split_idx = text.find("<Dialogue>", split_idx)
train_text = text[:split_idx]
val_text = text[split_idx:]


tokenizer = BPETokenizer(train_text)
train_data = tokenizer.encode(train_text)
val_data = tokenizer.encode(val_text)


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
                _ , loss = model(batch_ids, target_ids)

                split_losses.append(loss.item())

            losses[split] = torch.tensor(split_losses).mean()

    # Put model back into training mode
    model.train()

    return losses

model = TransformerModel(dropout, n_embd, n_head, n_layer, label_smoothing).to(device)
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

min_val_loss = float('inf')
best_iter = 0
best_train_loss = float('inf')

for iter in range(max_iters+1):  

    if iter % eval_interval == 0:
        losses = estimate_loss(model)

        print(
            f"Iteration {iter}, "
            f"Training loss = {losses['train']:.4f}, "
            f"Validation loss = {losses['val']:.4f}\n"
        )

        if losses['val'] < min_val_loss:
            min_val_loss = losses['val'].item()
            best_train_loss = losses['train'].item()
            best_iter = iter

            torch.save(model.state_dict(), 'checkpoints/best_model.pt')

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


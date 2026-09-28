import torch

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 320
n_head = 8
n_layer = 4
dropout = 0.40
learning_rate = 3e-4
vocab_size = 1000
dialogue_id = 256
weight_decay = 0.05
warmup_iter = 250
label_smoothing = 0.10
dialogueStr = '<Dialogue>'

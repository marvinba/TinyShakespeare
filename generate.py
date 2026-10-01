import torch
from model import TransformerModel
from config import dialogue_id, device, dropout, n_embd, n_head, n_layer, label_smoothing
from tokenizer import BPETokenizer 

best_model = TransformerModel(dropout, n_embd, n_head, n_layer, label_smoothing).to(device)
best_model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device, weights_only=True))
best_model.eval()

best_tokenizer = BPETokenizer()
best_tokenizer.load('checkpoints/tokenizer.json')
            
genIds = best_model.generate(torch.tensor([[dialogue_id]],dtype=torch.long,device=device))
genTokens = [
    best_tokenizer.getToken(token_id)
    for token_id in genIds[0].tolist()
    if token_id != dialogue_id
]

generated_bytes = b''.join(genTokens)
generated_text = generated_bytes.decode('utf-8', errors='replace')
generated_text = generated_text + '\n'

print(generated_text)
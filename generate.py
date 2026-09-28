from model import TransformerModel
from config import dialogue_id, device, dropout, n_embd, n_head, n_layer, label_smoothing
import torch
from tokenizer import BPETokenizer as tokenizer

best_model = TransformerModel(dropout, n_embd, n_head, n_layer, label_smoothing).to(device)
best_model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=True))
best_model.eval()
            
genIds = best_model.generate(torch.tensor([[dialogue_id]],dtype=torch.long,device=device))
genTokens = [
    tokenizer.getToken(token_id)
    for token_id in genIds[0].tolist()
    if token_id != dialogue_id
]

generated_bytes = b''.join(genTokens)
generated_text = generated_bytes.decode('utf-8', errors='replace')
generated_text = generated_text + '\n'

print(generated_text)

print(
    f"\nBest checkpoint: iteration {best_iter}\n"
    f"Training loss = {best_train_loss:.4f}\n"
    f"Validation loss = {min_val_loss:.4f}"
)
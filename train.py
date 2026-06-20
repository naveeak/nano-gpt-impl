import torch
from torch.optim import AdamW
from config import batch_size, block_size, max_iters, eval_interval, learning_rate, device
from config import eval_iters, n_embd, dropout, n_head, n_layer
from data import get_batch, vocab_size, encode, decode, stoi, itos
from model import BigramLanguageModel
import os

model = BigramLanguageModel(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

for iter_num in range(max_iters):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        xb, yb = get_batch('val')
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            _, val_loss = model(xb, yb)
        print(f"iter {iter_num}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'dropout': dropout,
    },
    'stoi': stoi,
    'itos': itos,
}
os.makedirs('checkpoints', exist_ok=True)
torch.save(checkpoint, 'checkpoints/model_step5000.pt')
print('Saved checkpoint to checkpoints/model_step5000.pt')

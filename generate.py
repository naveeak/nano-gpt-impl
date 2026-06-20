import torch
from huggingface_hub import hf_hub_download
from model import BigramLanguageModel

REPO_ID = "your-username/shakespeare-gpt"
FILENAME = "model_step5000.pt"

def load_model(device='cpu'):
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint['config']

    model = BigramLanguageModel(**cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    stoi, itos = checkpoint['stoi'], checkpoint['itos']
    decode = lambda l: ''.join(itos[i] for i in l)
    encode = lambda s: [stoi[c] for c in s]
    return model, encode, decode

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, encode, decode = load_model(device)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import matplotlib.pyplot as plt
from models import VanillaRNNModule, BLSTMModule, RNNAttention

# function to load the dataset of names, create mapping for embeddings and prepare data for training.
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        names = f.read().splitlines()
    
    # Character vocabulary
    chars = set(''.join(names))
    chars.add('<pad>')
    chars.add('<bos>') # Beginning of Sequence
    chars.add('<eos>') # End of sequence
    
    char2idx = {ch: i for i, ch in enumerate(sorted(chars))}
    idx2char = {i: ch for ch, i in char2idx.items()}
    
    return names, char2idx, idx2char

# function to convert a tensor of character indices to a string name.
def tensor_to_name(tensor, idx2char):
    name = ""
    for idx in tensor:
        # Check if idx is a tensor with .item() or already an int
        val = idx.item() if hasattr(idx, 'item') else idx
        ch = idx2char[val]
        if ch == '<eos>': break
        if ch not in ('<pad>', '<bos>'):
            name += ch
    return name
# function to train the given model on dataset
def train_model(model, data, char2idx, epochs=5, lr=0.005, save_name=None):
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f'--- Training {model.__class__.__name__} ---')
    print(f'Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 32
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        random.shuffle(data)

        for i in range(0, len(data), batch_size):
            batch_names = data[i:i+batch_size]
            max_len = max(len(n) for n in batch_names) + 2 

            input_seqs = torch.full((len(batch_names), max_len-1), char2idx['<pad>'], dtype=torch.long)
            target_seqs = torch.full((len(batch_names), max_len-1), char2idx['<pad>'], dtype=torch.long)
            for j, name in enumerate(batch_names):
                encoded = [char2idx['<bos>']] + [char2idx[c] for c in name] + [char2idx['<eos>']]
                input_seqs[j, :len(encoded)-1] = torch.tensor(encoded[:-1])
                target_seqs[j, :len(encoded)-1] = torch.tensor(encoded[1:])

            input_seqs, target_seqs = input_seqs.to(device), target_seqs.to(device)
            optimizer.zero_grad()
            outputs, _ = model(input_seqs)

            loss = criterion(outputs.transpose(1, 2), target_seqs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(data)/batch_size)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}')
        
    if save_name:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/{save_name}.pt')
        print(f'Model saved to checkpoints/{save_name}.pt')

    return model, epoch_losses

# function to generate sample name from trained models
def generate_sample(model, char2idx, idx2char, max_len=20, device='cpu'):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[char2idx['<bos>']]]).to(device)
        generated = []
        hidden = None
        
        for _ in range(max_len):
            out, hidden = model(x, hidden)
            # Take last output
            logits = out[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # Simple sampling
            next_char_idx = torch.multinomial(probs, 1).item()
            generated.append(next_char_idx)
            
            if idx2char[next_char_idx] == '<eos>':
                break
                
            x = torch.cat([x, torch.tensor([[next_char_idx]]).to(device)], dim=1)
            
        return tensor_to_name(generated, idx2char)

# function to evaluate the trained models generated names based on diversity and novelty
def evaluate(model, data, char2idx, idx2char, num_samples=100, device='cpu'):
    training_set = set(data)
    
    generated_names = []
    for _ in range(num_samples):
        name = generate_sample(model, char2idx, idx2char, device=device)
        generated_names.append(name.strip())
        
    unique_names = set(generated_names)
    novel_names = [n for n in unique_names if n not in training_set and len(n) > 2]
    
    diversity = len(unique_names) / num_samples if num_samples > 0 else 0
    novelty = len(novel_names) / len(unique_names) if len(unique_names) > 0 else 0
    
    return diversity, novelty, generated_names[:10]

if __name__ == "__main__":
    if not os.path.exists("names_list.txt"):
        print("names_list.txt not found. Please ensure the file is present.")
        exit(1)

    names, char2idx, idx2char = load_data("names_list.txt")
    vocab_size = len(char2idx)
    hidden_dim = 64
    
    models_dict = {
        "VanillaRNN": VanillaRNNModule(vocab_size, hidden_dim, n_layers=1),
        "BLSTM": BLSTMModule(vocab_size, hidden_dim),
        "RNN_Attention": RNNAttention(vocab_size, hidden_dim)
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    all_losses = {}
    for name, model in models_dict.items():
        trained_model, epoch_losses = train_model(model, names, char2idx, epochs=10, lr=0.005, save_name=name)
        all_losses[name] = epoch_losses
        m_diversity, m_novelty, samples = evaluate(trained_model, names, char2idx, idx2char, num_samples=100, device=device)
        results[name] = {
            "diversity": m_diversity,
            "novelty": m_novelty,
            "samples": samples
        }
    plt.figure(figsize=(10, 6))
    for model_name, losses in all_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=model_name, marker="o")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    print("\nSaved training loss curves to loss_curves.png")
    print("\n\n=== EVALUATION REPORT ===")
    for name, res in results.items():
        print(f"\nModel: {name}")
        print("Diversity Rate: {:.2%}".format(res["diversity"]))
        print("Novelty Rate:   {:.2%}".format(res["novelty"]))
        print(f"Sample Names generated: {', '.join(res['samples'])}")

# Import libraries
import torch
import torch.nn as nn

# Define the SAME model class again
# class MiniGPT(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, block_size):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.position_embedding = nn.Embedding(2 * block_size, embedding_dim)
#         self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
#         self.feedforward = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim * 4),
#             nn.ReLU(),
#             nn.Linear(embedding_dim * 4, embedding_dim)
#         )
#         self.layernorm1 = nn.LayerNorm(embedding_dim)
#         self.layernorm2 = nn.LayerNorm(embedding_dim)
#         self.output_layer = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, x):
#         batch_size, seq_len = x.shape
#         token_emb = self.token_embedding(x)
#         position_ids = torch.arange(seq_len, device=x.device)
#         position_emb = self.position_embedding(position_ids)
#         x = token_emb + position_emb
#         mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
#         mask = mask.masked_fill(mask == 1, float('-inf'))
#         attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
#         x = self.layernorm1(x + attn_out)
#         ff_out = self.feedforward(x)
#         x = self.layernorm2(x + ff_out)
#         logits = self.output_layer(x)
#         return logits

# class MiniGPT(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, block_size):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

#         # ✅ FIX: Large enough positional embedding
#         self.position_embedding = nn.Embedding(1024, embedding_dim)

#         self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
#         self.feedforward = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim * 4),
#             nn.ReLU(),
#             nn.Linear(embedding_dim * 4, embedding_dim)
#         )
#         self.layernorm1 = nn.LayerNorm(embedding_dim)
#         self.layernorm2 = nn.LayerNorm(embedding_dim)
#         self.output_layer = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, x):
#         batch_size, seq_len = x.shape
#         token_emb = self.token_embedding(x)
#         position_ids = torch.arange(seq_len, device=x.device)

#         # This will now NOT crash, since 1024 > 50 easily
#         position_emb = self.position_embedding(position_ids)

#         x = token_emb + position_emb
#         mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
#         mask = mask.masked_fill(mask == 1, float('-inf'))
#         attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
#         x = self.layernorm1(x + attn_out)
#         ff_out = self.feedforward(x)
#         x = self.layernorm2(x + ff_out)
#         logits = self.output_layer(x)
#         return logits

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 👇 Match exactly how it was during training
        self.position_embedding = nn.Embedding(2 * block_size, embedding_dim)
        
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)
        position_ids = torch.arange(seq_len, device=x.device)
        position_emb = self.position_embedding(position_ids)
        
        x = token_emb + position_emb
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.layernorm1(x + attn_out)
        
        ff_out = self.feedforward(x)
        x = self.layernorm2(x + ff_out)
        
        logits = self.output_layer(x)
        return logits


# Recreate the model
vocab = [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
vocab_size = len(vocab)
embedding_dim = 32
block_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MiniGPT(vocab_size, embedding_dim, block_size).to(device)

# Load the trained weights
model.load_state_dict(torch.load("mini_gpt.pth", map_location=device))

print("\n✅ Mini-GPT model loaded!")


# INFER

# Helper mappings
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# Start prompt
prompt = "h"   # starting letter
prompt = "w"   # starting letter
context = torch.tensor([[char_to_idx[c] for c in prompt]], dtype=torch.long).to(device)

model.eval()

# Generate 50 more characters
generated = prompt

# for _ in range(50):
#     logits = model(context)
    
#     # Get the logits of the last token
#     logits = logits[:, -1, :]
    
#     # Apply softmax to get probabilities
#     probs = torch.softmax(logits, dim=-1)
    
#     # Sample from the distribution
#     next_id = torch.multinomial(probs, num_samples=1).item()
    
#     # Add predicted token to generated text
#     generated += idx_to_char[next_id]
    
#     # Append new token to context
#     context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)

# Generate 15 more characters (context length must stay <= 16)
max_new_tokens = 15

for _ in range(max_new_tokens):
    logits = model(context)
    
    # Get logits of the last token
    logits = logits[:, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    
    generated += idx_to_char[next_id]
    
    # Append new token
    context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)
    
    # ✅ If context length > 16, keep only the last 16 tokens
    if context.shape[1] > 16:
        context = context[:, -16:]

print(f"\n📝 Generated text for prompt:{prompt} :\n")
print(generated)


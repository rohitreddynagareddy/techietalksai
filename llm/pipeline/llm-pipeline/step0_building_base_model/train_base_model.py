import torch
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# === Step 1: Prepare dataset ===
# tiny_corpus = [
#     "Hello world! This is a new model.",
#     "Machine learning is fun and full of possibilities.",
#     "Transformers are powerful tools for language generation.",
#     "I love open source projects and contributions.",
#     "Python is a versatile language used for many applications.",
#     "AI is revolutionizing industries like healthcare, finance, and education.",
#     "Large Language Models are trained on diverse internet text.",
#     "Natural language processing helps computers understand human language.",
#     "Fine-tuning improves model performance on specific tasks.",
#     "Neural networks learn from patterns in data through backpropagation."
# ]

# === Step 1: Load corpus from file ===
with open("corpus.txt", "r") as f:
    tiny_corpus = [line.strip() for line in f if line.strip()]
print(f"📚 Loaded {len(tiny_corpus)} lines from corpus.txt")

# === Step 2: Tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# === Step 3: Create dataset ===
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.examples = [
            tokenizer(t, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"].squeeze()
            for t in texts
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx], self.examples[idx]

dataset = TextDataset(tiny_corpus, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# === Step 4: Model Config and Init ===
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=64,
    n_ctx=64,
    n_embd=128,
    n_layer=4,
    n_head=4,
    pad_token_id=tokenizer.pad_token_id
)
model = GPT2LMHeadModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# === Step 5: Training Loop ===
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
for epoch in range(20):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        input_ids, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# === Step 6: Save Model ===
SAVE_DIR = "./tiny_gpt2_model"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("✅ Model and tokenizer saved to ./tiny_gpt2_model")
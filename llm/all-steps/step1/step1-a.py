import torch.multiprocessing
import time
import os
from datetime import timedelta
torch.multiprocessing.freeze_support()

# Checkpoint configuration
CHECKPOINT_DIR = "checkpoints"
SAVE_INTERVAL = 100  # Save every 100 batches
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(epoch, batch_idx, model, optimizer, loss, is_apple):
    """Save training state with Apple Silicon compatibility"""
    checkpoint = {
        'epoch': epoch,
        'batch': batch_idx,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
        'config': config.to_dict()
    }
    
    # Convert MPS tensors to CPU
    if is_apple:
        checkpoint = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                     for k, v in checkpoint.items()}
    
    path = f"{CHECKPOINT_DIR}/checkpoint_epoch{epoch}_batch{batch_idx}.pt"
    torch.save(checkpoint, path)
    print(f"💾 Saved checkpoint to {path}")
    return path

def load_checkpoint(checkpoint_path, model, optimizer, is_apple):
    """Load training state with device handling"""
    checkpoint = torch.load(checkpoint_path, 
                          map_location='mps' if is_apple else None)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # Move tensors to MPS if needed
    if is_apple:
        model.to('mps')
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to('mps')
    
    return checkpoint['epoch'], checkpoint['batch'], checkpoint['loss']

if __name__ == '__main__':
    # [Previous setup code remains identical until training loop]
    print("# Step 1: Install Required Libraries")
    print("pip install torch transformers tiktoken datasets")

    print("# Step 2: Import Necessary Modules")
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2LMHeadModel, GPT2Config
    import tiktoken
    from datasets import load_dataset

    print("# Step 3: Initialize Tokenizer")
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token
    vocab_size = enc.n_vocab

    print("# Step 4: Load and Prepare Dataset")
    # The script uses the **WikiText-2 “raw” v1** dataset (from Hugging Face’s Datasets library) 
    # as its training corpus. Specifically, it calls:

    # – **What is WikiText-2 Raw v1?**  
    #   A language-modeling benchmark comprising full, unprocessed Wikipedia articles (i.e. 
    #   keeping case, punctuation, numbers), with splits of 36 718 train examples, 3 760 validation examples, and 4 358 test examples.  ([Salesforce/wikitext · Datasets at Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext?utm_source=chatgpt.com))  
    # – **Why use it?**  
    #   It’s substantially larger than older benchmarks like PTB, preserves long-range 
    #   dependencies, and is widely used for causal-LM pretraining and fine-tuning experiments.  
    #   ([WikiText-2 Dataset - Papers With Code](https://paperswithcode.com/dataset/wikitext-2?utm_source=chatgpt.com))

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    block_size = 512

    def process_texts(examples):
        all_tokens = []
        for text in examples["text"]:
            if text.strip():
                tokens = enc.encode_ordinary(text)
                tokens.append(eot_token)
                all_tokens.extend(tokens)
        chunks = [all_tokens[i:i+block_size] 
                for i in range(0, len(all_tokens), block_size)]
        chunks = [c for c in chunks if len(c) == block_size]
        return {"input_ids": chunks}

    tokenized_dataset = dataset.map(
        process_texts,
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )

    print("# Step 5: Create PyTorch Dataset")
    class GPTDataset(Dataset):
        def __init__(self, hf_dataset):
            self.examples = []
            for example in hf_dataset["input_ids"]:
                self.examples.extend(example)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return torch.tensor(self.examples[idx], dtype=torch.long)

    train_dataset = GPTDataset(tokenized_dataset["train"])

    print("# Step 6: Initialize Model")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)

    # Device detection
    is_apple_silicon = torch.backends.mps.is_available()
    device = "mps" if is_apple_silicon else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🍎 Using device: {device}")
    model.to(device)

    print("# Step 7: Set Up Training Components")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler(enabled=True) if is_apple_silicon else None

    # DataLoader configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0 if is_apple_silicon else 4,
        pin_memory=not is_apple_silicon
    )
    num_epochs = 3
    print(f"\n🚀 Starting training for {num_epochs} epochs")
    # Modified Training Loop
    start_epoch = 0
    start_batch = 0

    # Add this before training starts (after model/optimizer setup)
    if os.path.exists(f"{CHECKPOINT_DIR}/latest_checkpoint.pt"):
        resume = input("Found existing checkpoint. Resume training? (y/n): ")
        if resume.lower() == 'y':
            start_epoch, start_batch, _ = load_checkpoint(
                f"{CHECKPOINT_DIR}/latest_checkpoint.pt",
                model,
                optimizer,
                is_apple_silicon
            )
            print(f"Resuming from epoch {start_epoch+1} batch {start_batch}")

    print("# Step 8: Training Loop")

    total_steps = len(train_loader)
    print(f"Total batches per epoch: {total_steps}")

    model.train()
    start_time = time.perf_counter()
    # Modified training loop    
    # Modified training loop with proper checkpoint handling
    start_epoch = 0
    start_batch = 0

    # Checkpoint resuming logic
    latest_path = f"{CHECKPOINT_DIR}/latest_checkpoint.pt"
    if os.path.exists(latest_path):
        resume = input("Found existing checkpoint. Resume training? (y/n): ")
        if resume.lower() == 'y':
            start_epoch, start_batch, _ = load_checkpoint(
                latest_path,
                model,
                optimizer,
                is_apple_silicon
            )
            print(f"Resuming from epoch {start_epoch+1} batch {start_batch}")

    # Training loop
    # for epoch in range(start_epoch, num_epochs):
    #     epoch_start = time.time()
    #     total_loss = 0
    #     batch_count = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue  # Skip processed batches

            inputs = batch.to(device)
            
            # Training steps
            if is_apple_silicon:
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(inputs, labels=inputs)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            total_loss += loss.item()
            batch_count += 1

            # Checkpoint saving logic
            if (batch_idx + 1) % SAVE_INTERVAL == 0:
                checkpoint_path = save_checkpoint(
                    epoch, batch_idx, model, optimizer, loss.item(), is_apple_silicon
                )
                # Update latest checkpoint using symlink
                if os.path.lexists(latest_path):
                    os.remove(latest_path)
                os.symlink(os.path.basename(checkpoint_path), latest_path)

            # Progress reporting
            if batch_idx % 100 == 0:
                avg_loss = total_loss / batch_count
                now = time.perf_counter()
                elapsed = now - start_time
                batches_done = epoch * total_steps + batch_idx
                avg_time_per_batch = elapsed / (batches_done + 1)
                batches_left = num_epochs * total_steps - (batches_done + 1)
                eta_total = avg_time_per_batch * batches_left
                eta_epoch = avg_time_per_batch * (total_steps - batch_idx - 1)
                
                print(f"Epoch {epoch+1} | Batch {batch_idx} | "
                      f"Loss {loss.item():.4f} | "
                      f"Time/Batch {avg_time_per_batch:.2f}s | "
                      f"ETA this epoch: {eta_epoch/60:.1f}m | "
                      f"ETA total: {eta_total/3600:.1f}h")

        # End of epoch processing
        checkpoint_path = save_checkpoint(
            epoch, len(train_loader)-1, model, optimizer, loss.item(), is_apple_silicon
        )
        if os.path.lexists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(checkpoint_path), latest_path)

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = total_loss / len(train_loader)
        print(f"\n⏰ Epoch {epoch+1}/{num_epochs} completed in {epoch_time/60:.2f} minutes")
        print(f"📉 Average Loss: {avg_loss:.4f}")
        
        if is_apple_silicon:
            torch.mps.empty_cache()

    print("# Step 9: Save Model")
    torch.save(model.state_dict(), "gpt2_pretrained.pth")
    print("\n✅ Training completed successfully!")
import torch.multiprocessing
torch.multiprocessing.freeze_support()
import time

if __name__ == '__main__':
    print("# Step 1: Install Required Libraries")
    print("pip install torch transformers tiktoken datasets")

    print("# Step 2: Import Necessary Modules")
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2LMHeadModel, GPT2Config
    import tiktoken
    from datasets import load_dataset
    import platform

    print("# Step 3: Initialize Tokenizer")
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token
    vocab_size = enc.n_vocab

    print("# Step 4: Load and Prepare Dataset")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    block_size = 512

    def process_texts(examples):
        all_tokens = []
        for text in examples["text"]:
            if text.strip():
                toks = enc.encode_ordinary(text)
                toks.append(eot_token)
                all_tokens.extend(toks)
        # chunk into fixed-length blocks
        chunks = [
            all_tokens[i : i + block_size]
            for i in range(0, len(all_tokens), block_size)
        ]
        # only full-length blocks
        return {"input_ids": [c for c in chunks if len(c) == block_size]}

    tokenized = dataset.map(
        process_texts,
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )

    print("# Step 5: Create PyTorch Dataset")
    class GPTDataset(Dataset):
        def __init__(self, hf_split):
            # hf_split["input_ids"] is List[List[int]]
            self.examples = hf_split["input_ids"]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return torch.tensor(self.examples[idx], dtype=torch.long)

    train_dataset = GPTDataset(tokenized["train"])

    print("# Step 6: Initialize Model and Device")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)

    machine = platform.machine()
    use_mps = (machine in ["arm64", "aarch64"]) and torch.backends.mps.is_available()
    device = "mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Machine: {machine} → Using device: {device}")

    model.to(device)

    print("# Step 7: Set Up Training Components")
    # Shared DataLoader settings
    loader_kwargs = dict(
        batch_size=8,
        shuffle=True,
        num_workers=0 if use_mps else 2,
        pin_memory=False if use_mps else True,
        persistent_workers=False
    )
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5, fused=use_mps)
    scaler    = torch.amp.GradScaler(enabled=use_mps)

    num_epochs = 3

    print("# Step 8: Training Loop")


    total_steps = len(train_loader)
    print(f"Total batches per epoch: {total_steps}")

    model.train()
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch.to(device, non_blocking=use_mps)

            autocast_ctx = torch.autocast(
                device_type="mps" if use_mps else "cpu",
                dtype=torch.float16
            ) if use_mps else torch.cpu.amp.autocast()

            with autocast_ctx:
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss

            if use_mps:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                # now = time.perf_counter()
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

                now = time.perf_counter()
                elapsed = now - start_time
                batches_done = epoch * total_steps + batch_idx
                avg_time_per_batch = elapsed / (batches_done + 1)
                batches_left = num_epochs * total_steps - (batches_done + 1)
                eta_total = avg_time_per_batch * batches_left
                eta_epoch = avg_time_per_batch * (total_steps - batch_idx - 1)
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Loss {loss.item():.4f} | "
                    f"Time/Batch {avg_time_per_batch:.2f}s | "
                    f"ETA this epoch: {eta_epoch/60:.1f}m | "
                    f"ETA total: {eta_total/3600:.1f}h"
                )

            if use_mps:
                torch.mps.empty_cache()

        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Average Loss: {avg:.4f}")

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        print(f"→ Finished epoch {epoch} in {epoch_time/60:.2f} minutes")

    total_time = time.perf_counter() - start_time
    print(f"All {num_epochs} epochs completed in {total_time/60:.2f} minutes")

    print("# Step 9: Save Model")
    torch.save(model.state_dict(), "gpt2_pretrained.pth")

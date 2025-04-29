import torch.multiprocessing
torch.multiprocessing.freeze_support()


if __name__ == '__main__':

  print ("# Step 1: Install Required Libraries")

  print ("pip install torch transformers tiktoken datasets")

  print ("# Step 2: Import Necessary Modules")

  import torch
  from torch.utils.data import Dataset, DataLoader
  from transformers import GPT2LMHeadModel, GPT2Config
  import tiktoken
  from datasets import load_dataset

  print ("# Step 3: Initialize Tokenizer")

  # Initialize tiktoken tokenizer (GPT-2's BPE tokenizer)
  enc = tiktoken.get_encoding("gpt2")
  eot_token = enc.eot_token  # End-of-text token
  vocab_size = enc.n_vocab    # 50257 for GPT-2

  print ("# Step 4: Load and Prepare Dataset")

  # Load WikiText dataset
  dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

  # Define sequence length (same as GPT-2's context window)
  block_size = 512

  def process_texts(examples):
      """Tokenize and split into chunks of block_size"""
      all_tokens = []
      for text in examples["text"]:
          if text.strip():  # Skip empty texts
              tokens = enc.encode_ordinary(text)
              tokens.append(eot_token)  # Add EOS token
              all_tokens.extend(tokens)
      
      # Split into blocks
      chunks = [all_tokens[i:i+block_size] 
                for i in range(0, len(all_tokens), block_size)]
      chunks = [c for c in chunks if len(c) == block_size]
      
      return {"input_ids": chunks}

  # Process dataset in batches
  tokenized_dataset = dataset.map(
      process_texts,
      batched=True,
      batch_size=1000,
      remove_columns=["text"]
  )

  print ("# Step 5: Create PyTorch Dataset")

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
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

  print("# Step 6: Initialize Model")

  # Match GPT-2 small architecture
  config = GPT2Config(
      vocab_size=vocab_size,
      n_positions=block_size,
      n_embd=768,
      n_layer=12,
      n_head=12
  )


  model = GPT2LMHeadModel(config)

  import platform
  machine_type = platform.machine()
  is_apple_silicon = machine_type in ["arm64", "aarch64"]
  print(f"🖥️ Machine type: {machine_type}")
  print(f"🍎 Running on Apple Silicon: {is_apple_silicon}")
  # Check if PyTorch is using the MPS backend
  if is_apple_silicon:
      if torch.backends.mps.is_available():
          device = "mps"
          print("🚀 PyTorch is using the MPS backend for Apple Silicon.")
      else:
          device = "cpu"
          print("⚠️ PyTorch is not using the MPS backend. Falling back to CPU.")
  else:
      #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      device = "cuda" if torch.cuda.is_available() else "cpu"
      print(f"🖥️ Using device: {device}")

  # Initialize model on MPS
  torch.device(device)
  model.to(device)

  print("# Step 7: Set Up Training Components")

  apple = True
  # CPU/GPU
  if not apple:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

  else:
    # M4 ----------
    # Enable Metal-optimized DataLoader
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True  # Critical for MPS performance
    # )
    # Change your DataLoader to:
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing on macOS temporarily
        pin_memory=False  # Disable pin memory for MPS
    )
    from torch.optim import AdamW
    # Use Apple-optimized AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5, fused=True)
    # Mixed precision training
    # scaler = torch.cuda.amp.GradScaler()  # Works with MPS
    # With MPS-compatible version:
    scaler = torch.amp.GradScaler()  # New MPS-compatible scaler
    # scaler = torch.amp.GradScaler(device_type='mps')  # New MPS-compatible scaler
    # M4 ----------


  num_epochs = 3

  print("# Step 8: Training Loop")

  model.train()

  if apple:
    # M4--------- ADDITION
    # model.train()
    # for batch in train_loader:
    #     inputs = batch.to(device, non_blocking=True)
        
    #     with torch.autocast(device_type='mps', dtype=torch.float16):
    #         outputs = model(inputs, labels=inputs)
    #         loss = outputs.loss
        
    #     # Metal-optimized backprop
    #     scaler.scale(loss).backward()
    #     scaler.step(optimizer)
    #     scaler.update()
    #     optimizer.zero_grad(set_to_none=True)  # Reduce memory usage
    # Initialize model and optimizer
    model = GPT2LMHeadModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler(enabled=True)

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs = batch.to(device, non_blocking=True)
            
            # MPS-compatible mixed precision
            with torch.autocast(device_type='mps', dtype=torch.float16):
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
            
            # Scaled backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Memory cleanup
            torch.mps.empty_cache()
    # M4---------------

  else:
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch.to(device)
            
            # Forward pass with labels for auto-regressive LM
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")


  print("# Step 9: Save Model")
  torch.save(model.state_dict(), "gpt2_pretrained.pth")


  print("""

  ### Key Components Explained:

  1. **Tokenizer (tiktoken)**:
     - Implements GPT-2's BPE tokenization
     - Handles text splitting and special tokens
     - `encode_ordinary()` converts text to token IDs

  2. **Dataset Processing**:
     - Splits text into fixed-length sequences (512 tokens)
     - Adds EOS tokens between documents
     - Creates batches of contiguous text sequences

  3. **Model Architecture**:
     - Uses standard GPT-2 configuration
     - 12 transformer layers with 12 attention heads
     - 768-dimensional embeddings
     - Language modeling head for next-token prediction

  4. **Training Dynamics**:
     - Cross-entropy loss on shifted predictions
     - AdamW optimizer with standard learning rate
     - Full sequence processing (no padding needed)

  ### Important Considerations:

  1. **Data Efficiency**:
     - Use larger datasets (BookCorpus, OpenWebText) for better results
     - Consider streaming datasets for large-scale training

  2. **Training Optimization**:
     - Add gradient accumulation for larger batch sizes
     - Implement learning rate scheduling
     - Use mixed-precision training (`torch.cuda.amp`)

  3. **Evaluation**:
     - Monitor perplexity on validation set
     - Add periodic generation samples
     - Track memory usage and throughput

  This implementation provides a simplified but complete workflow for pretraining a GPT-style model. The actual training would require significant computational resources (multiple GPUs and days of training time) to achieve meaningful results.

  """)


  # if __name__ == '__main__':
  #     # Your training code here

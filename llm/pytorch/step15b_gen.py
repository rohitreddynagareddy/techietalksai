# # step15_tiktoken_pretrain_gpt2_from_scratch.py

# import os
import torch
# from datasets import load_dataset
# import tiktoken
# from transformers import (
#     PreTrainedTokenizerFast,
#     GPT2Config,
#     GPT2LMHeadModel,
#     DataCollatorForLanguageModeling,
#     Trainer,
#     TrainingArguments,
# )
# from transformers.integrations.tiktoken import convert_tiktoken_to_fast

# # ── 1. Paths & Hyperparameters ───────────────────────────────────────────────
# CORPUS_PATH   = "story_corpus.txt"      # one-line-per-story text file
# TOKEN_DIR     = "tokenizer/"            # where tokenizer.json will live
# OUTPUT_DIR    = "gpt2-output/"          # final model+tokenizer checkpoint
# BLOCK_SIZE    = 256                      # sequence length
# BATCH_SIZE    = 4                        # per-device batch size
# EPOCHS        = 3                        # number of pretraining epochs
# LEARNING_RATE = 3e-4                     # learning rate

# # ── 2. Create & Convert tiktoken → 🤗 fast tokenizer ─────────────────────────
# enc = tiktoken.get_encoding("gpt2")     # load GPT-2 byte‑level BPE :contentReference[oaicite:3]{index=3}
# os.makedirs(TOKEN_DIR, exist_ok=True)
# convert_tiktoken_to_fast(enc, TOKEN_DIR)  # writes tokenizer.json :contentReference[oaicite:4]{index=4}

# # ── 3. Load HF Tokenizer & Define pad_token ─────────────────────────────────
# # tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKEN_DIR)
# # OPTION 1: Reuse EOS as PAD
# # tokenizer.pad_token = tokenizer.eos_token  # assigns padding token :contentReference[oaicite:5]{index=5}

# # ── 3. Load HF Tokenizer & Define Special Tokens ────────────────────────────
# tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKEN_DIR)

# # Ensure special tokens are set (critical for GPT-2 compatibility)
# tokenizer.add_special_tokens({
#     "eos_token": "<|endoftext|>",  # Explicitly set EOS
#     "pad_token": "<|endoftext|>"   # Use EOS as PAD (common for causal LM)
# })

# # Verify tokens are correctly assigned
# assert tokenizer.eos_token == "<|endoftext|>"
# assert tokenizer.pad_token == "<|endoftext|>"


# # OPTION 2: Add a distinct [PAD] token instead (uncomment if preferred)
# # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # adds new token :contentReference[oaicite:6]{index=6}
# # model.resize_token_embeddings(len(tokenizer))

# # ── 4. Build GPT‑2 Config & Model ─────────────────────────────────────────────
# config = GPT2Config(
#     vocab_size=tokenizer.vocab_size,
#     n_positions=BLOCK_SIZE,
#     n_ctx=BLOCK_SIZE,
#     n_embd=256,
#     n_layer=6,
#     n_head=8,
# )
# model = GPT2LMHeadModel(config)

# # ── 5. Load & Tokenize Dataset ────────────────────────────────────────────────
# # raw_ds = load_dataset("text", data_files={"train": CORPUS_PATH})
# # def tokenize_and_group(examples):
# #     ids = tokenizer(examples["text"],
# #                     truncation=True,
# #                     max_length=BLOCK_SIZE)["input_ids"]
# #     total = (len(ids) // BLOCK_SIZE) * BLOCK_SIZE
# #     return {"input_ids": [ids[i : i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]}

# # tokenized = raw_ds["train"].map(
# #     tokenize_and_group,
# #     batched=False,
# #     remove_columns=["text"],
# # )

# # ── 5. Load & Tokenize Dataset ────────────────────────────────────────────────
# raw_ds = load_dataset("text", data_files={"train": CORPUS_PATH})

# def tokenize_and_group(examples):
#     # Tokenize all texts in batch (no truncation yet)
#     tokenized = tokenizer(examples["text"], truncation=False)
    
#     # Concatenate tokens from all texts in batch
#     concatenated_ids = [token for ids in tokenized["input_ids"] for token in ids]
    
#     # Truncate to multiple of BLOCK_SIZE
#     total_length = (len(concatenated_ids) // BLOCK_SIZE) * BLOCK_SIZE
#     truncated_ids = concatenated_ids[:total_length]
    
#     # Split into BLOCK_SIZE chunks
#     return {
#         "input_ids": [truncated_ids[i:i+BLOCK_SIZE] 
#                       for i in range(0, total_length, BLOCK_SIZE)]
#     }

# tokenized = raw_ds["train"].map(
#     tokenize_and_group,
#     batched=True,  # Critical change!
#     remove_columns=["text"],
#     batch_size=1000  # Process 1000 texts at a time
# )
# # ── 6. Data Collator ───────────────────────────────────────────────────────────
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,
#     pad_to_multiple_of=BLOCK_SIZE,  # pad length to BLOCK_SIZE
# )

# # ── 7. Training Arguments & Trainer ────────────────────────────────────────────
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     overwrite_output_dir=True,
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=2,
#     learning_rate=LEARNING_RATE,
#     weight_decay=0.01,
#     save_total_limit=2,
#     save_steps=500,
#     logging_steps=100,
#     fp16=torch.cuda.is_available(),
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized,
#     data_collator=data_collator,
# )

# # ── 8. Launch Pretraining ─────────────────────────────────────────────────────
# if __name__ == "__main__":
#     trainer.train()
#     trainer.save_model(OUTPUT_DIR)
#     tokenizer.save_pretrained(OUTPUT_DIR)
#     print(f"✅ Pretraining completed. Model & tokenizer saved to {OUTPUT_DIR}")

from transformers import pipeline
gen = pipeline(
    "text-generation",
    model="gpt2-output/",
    tokenizer="gpt2-output/",
    device=0 if torch.cuda.is_available() else -1,
)
print(gen("Once upon a time", max_length=100, do_sample=True, top_k=50))
print(gen("Our business is", max_length=100, do_sample=True, top_k=50))


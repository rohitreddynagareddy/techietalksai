# pretrain_gpt2_from_scratch.py

import os
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch

########################################
# 1. Train a BPE tokenizer on your stories
########################################

def train_tokenizer(corpus_file="story_corpus.txt",
                    vocab_size=30_000,
                    tokenizer_dir="tokenizer/"):
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    # trains on one file (one story per line)
    tokenizer.train(files=[corpus_file],
                    vocab_size=vocab_size,
                    min_frequency=2,
                    special_tokens=[
                        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
                    ])
    tokenizer.save_model(tokenizer_dir)
    print(f"Tokenizer saved in {tokenizer_dir}")

########################################
# 2. Build GPT‑2 config & model from scratch
########################################

def build_model_from_scratch(tokenizer_dir="tokenizer/",
                             config_kwargs=None,
                             model_dir="gpt2-scratch/"):
    # load tokenizer to inspect vocab size
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir,
                                                  max_len=512,
                                                  pad_token="<pad>")
    
    # default small GPT-2 config
    default_cfg = dict(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_ctx=512,
        n_embd=256,
        n_layer=6,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    cfg = GPT2Config(**(config_kwargs or default_cfg))
    
    # instantiate model with random init
    model = GPT2LMHeadModel(cfg)
    
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Empty GPT-2 model & tokenizer saved to {model_dir}")
    return model, tokenizer

########################################
# 3. Prepare dataset for Causal LM
########################################

def tokenize_and_group(examples, tokenizer, block_size=512):
    outputs = tokenizer(examples["text"],
                        truncation=True,
                        max_length=block_size,
                        return_special_tokens_mask=False)
    # HuggingFace Trainer expects input_ids only
    return {"input_ids": outputs["input_ids"]}

########################################
# 4. Pretrain with HuggingFace Trainer
########################################

def pretrain(corpus_file="story_corpus.txt",
             model_dir="gpt2-scratch/",
             output_dir="gpt2-scratch/",
             epochs=3,
             batch_size=4,
             block_size=512,
             lr=5e-4):
    
    # load the empty model + tokenizer you saved
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir,
                                                  pad_token="<pad>")
    
    # load raw text dataset
    ds = load_dataset("text", data_files={"train": corpus_file})
    
    # tokenize & group
    tokenized = ds["train"].map(
        lambda ex: tokenize_and_group(ex, tokenizer, block_size),
        batched=True,
        remove_columns=["text"]
    )
    
    # data collator for causal LM (no masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=block_size
    )
    
    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Pretrained model saved at {output_dir}")

########################################
# 5. Putting it all together
########################################

if __name__ == "__main__":
    CORPUS = "story_corpus.txt"     # your one-story-per-line file
    TOKEN_DIR = "tokenizer/"
    MODEL_DIR = "gpt2-scratch/"
    
    # 1. Train tokenizer
    train_tokenizer(corpus_file=CORPUS,
                    vocab_size=20_000,
                    tokenizer_dir=TOKEN_DIR)
    
    # 2. Build empty GPT-2
    build_model_from_scratch(tokenizer_dir=TOKEN_DIR,
                             config_kwargs=None,
                             model_dir=MODEL_DIR)
    
    # 3. Pretrain
    pretrain(corpus_file=CORPUS,
             model_dir=MODEL_DIR,
             output_dir=MODEL_DIR,
             epochs=2,
             batch_size=2,
             block_size=256,
             lr=3e-4)

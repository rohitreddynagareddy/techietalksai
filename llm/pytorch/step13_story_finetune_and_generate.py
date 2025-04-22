# 2. story_finetune_and_generate.py

import os
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
import torch

def fine_tune_gpt2(
    story_file: str = "story_corpus.txt",
    output_dir: str = "gpt2-story-finetuned",
    epochs: int = 3,
    batch_size: int = 4,
    block_size: int = 512,
    learning_rate: float = 5e-5,
):
    # 2.1 Load your story corpus: one story per line
    dataset = load_dataset("text", data_files={"train": story_file})
    
    # 2.2 Initialize tokenizer & model
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # GPT-2 has no pad token by default
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    
    # 2.3 Tokenize & group into blocks
    def tokenize_and_group(examples):
        # Tokenize all texts
        tokens = tokenizer(examples["text"], truncation=True, max_length=block_size, return_special_tokens_mask=False)
        return tokens
    
    tokenized = dataset["train"].map(
        tokenize_and_group,
        batched=True,
        remove_columns=["text"],
    )
    
    # 2.4 Data collator for causal LM (i.e. GPT‑2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=block_size
    )
    
    # 2.5 Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        save_steps=500,
        save_total_limit=2,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),  # use mixed‑precision if available
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
    )
    
    # 2.6 Trainer    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    # 2.7 Train!
    trainer.train()
    
    # 2.8 Save final model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def generate_story(
    model_dir: str = "gpt2-story-finetuned",
    prompt: str = "Once upon a time",
    max_length: int = 200,
    num_return_sequences: int = 1,
):
    # 3.1 Load fine‑tuned model & tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    
    # 3.2 Prepare generation pipeline
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    
    # 3.3 Generate
    outputs = gen(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
    )
    return outputs

if __name__ == "__main__":
    # Fine‑tune on your story corpus
    fine_tune_gpt2(
        story_file="story_corpus.txt",
        output_dir="gpt2-story-finetuned",
        epochs=2,
        batch_size=2,
    )
    
    # Generate an example story
    examples = generate_story(
        model_dir="gpt2-story-finetuned",
        prompt="Deep in the enchanted forest, a lone traveler",
        max_length=150,
        num_return_sequences=2,
    )
    for i, out in enumerate(examples):
        print(f"\n=== GENERATED STORY #{i+1} ===\n{out['generated_text']}\n")

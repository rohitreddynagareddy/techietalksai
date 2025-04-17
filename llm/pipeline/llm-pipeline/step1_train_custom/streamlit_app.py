import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Constants
MODEL_DIR = "/app/output_model"

TINY_MODEL_DIR = "/app/tiny_gpt2_model"


st.set_page_config(page_title="LLM Mini Trainer & Inference", layout="wide")
st.title("🧠 Fine-Tune & Inference Demo (GPT-2)")

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load tokenizer and model if available, else fallback to GPT-2
@st.cache_resource
def load_model_and_tokenizer():
    # if os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
    # if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    #     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    #     model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).cpu()
    # else:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").cpu()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# --- TRAINING SECTION ---
st.header("🛠️ Training")

with st.form("train_form"):
    corpus_text = st.text_area("📚 Paste Your Corpus Text Below", height=200)
    epochs = st.number_input("⏱️ Number of Epochs", min_value=1, max_value=10, value=1)
    submit_train = st.form_submit_button("🚀 Train Now")

if submit_train and corpus_text.strip():
    with st.spinner("Training in progress..."):
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        inputs = tokenizer(corpus_text, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(torch.device("cpu"))
        input_ids = inputs["input_ids"]

        st.subheader("🔢 Encoded Tokens")
        st.code(input_ids[0].tolist(), language="json")

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        model.to(device)
        model.train()

        for epoch in range(epochs):
            outputs = model(input_ids.to(device), labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            st.info(f"✅ Epoch {epoch + 1} Loss: {loss.item():.4f}")

        # Save model & tokenizer
        # os.makedirs("output_model", exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        st.success("🎉 Training Complete and Saved to Disk!")

        # Force reload model
        st.cache_resource.clear()
        st.rerun()

# # --- INFERENCE SECTION ---
# st.header("🔍 Inference")

# prompt = st.text_input("✍️ Enter Your Prompt", "Once upon a time")
# max_tokens = st.slider("🧮 Max New Tokens to Generate", 10, 200, 50)

# if st.button("🔮 Generate"):
#     with st.spinner("Generating..."):
#         inputs = tokenizer(prompt, return_tensors="pt")
#         inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
#         output_ids = model.generate(**inputs, max_new_tokens=max_tokens)

#         st.subheader("🧱 Input Tokens")
#         st.code(inputs["input_ids"][0].tolist(), language="json")

#         st.subheader("🧱 Output Tokens")
#         st.code(output_ids[0].tolist(), language="json")

#         st.subheader("📝 Decoded Output")
#         st.success(tokenizer.decode(output_ids[0], skip_special_tokens=True))


# Define device (just like generate.py)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# @st.cache_resource
# def load_model_and_tokenizer():
#     if os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
#         st.info(f"📦 Loading fine-tuned model from `{MODEL_DIR}`")
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#         tokenizer.pad_token = tokenizer.eos_token
#         model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
#     else:
#         st.warning("⚠️ Fine-tuned model not found. Falling back to base GPT-2.")
#         tokenizer = AutoTokenizer.from_pretrained("gpt2")
#         tokenizer.pad_token = tokenizer.eos_token
#         model = AutoModelForCausalLM.from_pretrained("gpt2")
        
#     model.to(device)
#     model.eval()
#     return tokenizer, model

# @st.cache_resource
# def load_model_and_tokenizer(model_choice: str):
#     if model_choice == "Fine-tuned":
#         model_path = MODEL_DIR
#         label = "📦 Loading fine-tuned model from"
#     else:
#         model_path = "gpt2"
#         label = "📦 Loading base GPT-2 model from HuggingFace"

#     st.info(f"{label} `{model_path}`")

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.pad_token = tokenizer.eos_token
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     model.to(device)
#     model.eval()
#     return tokenizer, model

@st.cache_resource
def load_model_and_tokenizer(model_choice: str):
    if model_choice == "Fine-tuned":
        model_path = MODEL_DIR
        label = "📦 Loading fine-tuned model from"
    # elif model_choice == "Tiny GPT-2":
    #     model_path = TINY_MODEL_DIR
    #     label = "🧪 Loading tiny GPT-2 base model from"
    else:
        model_path = "gpt2"
        label = "📦 Loading base GPT-2 model from HuggingFace"

    st.info(f"{label} `{model_path}`")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


# st.sidebar.title("🔧 Model Settings")
# model_choice = st.sidebar.selectbox(
#     "Choose model to use for inference:",
#     ["Fine-tuned", "GPT-2"],
#     index=0 if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")) else 1
# )

# Decide available models based on presence
available_models = []
if os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    available_models.append("Fine-tuned")
# if os.path.exists(os.path.join(TINY_MODEL_DIR, "pytorch_model.bin")) or os.path.exists(os.path.join(TINY_MODEL_DIR, "model.safetensors")):
#     available_models.append("Tiny GPT-2")
available_models.append("GPT-2")

st.sidebar.title("🔧 Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose model to use for inference:",
    available_models,
    index=0
)


tokenizer, model = load_model_and_tokenizer(model_choice)

# tokenizer, model = load_model_and_tokenizer()

# --- INFERENCE SECTION ---
st.header("🔍 Inference.")

prompt = st.text_input("✍️ Enter Your Prompt", "I love")
max_tokens = st.slider("🧮 Max New Tokens to Generate", 1, 50, 3)

if st.button("🔮 Generate"):
    with st.spinner("Generating..."):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        st.subheader("🧱 Input Tokens")
        st.code(inputs["input_ids"][0].tolist(), language="json")

        st.subheader("🧱 Output Tokens")
        st.code(output_ids[0].tolist(), language="json")

        st.subheader("📝 Decoded Output")
        st.success(tokenizer.decode(output_ids[0], skip_special_tokens=True))


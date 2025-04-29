import streamlit as st
import numpy as np
import torch
import torch.nn as nn

st.title("Cross-Entropy Calculator")

# Inputs for true and predicted probabilities
y_true = st.text_input("True probabilities (comma-separated)", "0,0,1")
y_pred = st.text_input("Predicted probabilities (comma-separated)", "0.1,0.2,0.7")

# Convert inputs to arrays
y_true = np.array([float(x) for x in y_true.split(",")])
y_pred = np.array([float(x) for x in y_pred.split(",")])

if st.button("Calculate"):
    # Manual calculation
    manual_ce = -np.sum(y_true * np.log(y_pred + 1e-10))  # Avoid log(0)
    
    # PyTorch calculation
    ce_loss = nn.CrossEntropyLoss()
    torch_true = torch.from_numpy(y_true).float()
    torch_pred = torch.from_numpy(y_pred).float().unsqueeze(0)
    torch_ce = ce_loss(torch_pred, torch.argmax(torch_true).unsqueeze(0))
    
    st.write(f"Manual Cross-Entropy: {manual_ce:.4f}")
    st.write(f"PyTorch Cross-Entropy: {torch_ce.item():.4f}")

import streamlit as st
import numpy as np
import torch
import torch.nn as nn

st.title("Softmax Demo")

# Input for logits (raw model outputs)
logits_input = st.text_input("Enter logits (comma-separated)", "2.0, 1.0, 0.1")

# Convert input to array
logits = np.array([float(x) for x in logits_input.split(",")])

if st.button("Calculate Softmax"):
    # Manual softmax calculation
    exponents = np.exp(logits - np.max(logits))  # Numerical stability trick
    manual_softmax = exponents / np.sum(exponents)

    # PyTorch softmax calculation
    #torch_softmax = nn.Softmax(dim=-1)(torch.tensor(logits).numpy()
    torch_softmax = nn.Softmax(dim=-1)(torch.tensor(logits)).numpy()

    # Display results
    st.write("### Manual Softmax:")
    st.write(manual_softmax.round(4))
    
    st.write("### PyTorch Softmax:")
    st.write(torch_softmax.round(4))

    # Visualization
    st.bar_chart({
        "Manual": manual_softmax,
        "PyTorch": torch_softmax
    })

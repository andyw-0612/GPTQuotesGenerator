import numpy as np
import pandas as pd
import torch
import streamlit as st
import names
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

st.header("Quote Generator")
st.markdown("This uses the GPT-2 Model from Open AI")

col1, col2 = st.columns(2)
generated_max_len = col1.number_input("Max Length of Generated Text", value=1000)
num_outputs = col2.number_input("Number of Quotes to Generate", value=2)
MODEL_DIR = "SavedModel/"
DEVICE = "cpu"

def run_model(generated_max_len, num_outputs):
    if num_outputs <= 0 or generated_max_len <= 0:
        st.markdown("Please input a positive integer")
        return
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    prompt = "<|startoftext|>"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(DEVICE)
    sample_outputs = model.generate(
                                    generated, 
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = generated_max_len,
                                    top_p=0.95, 
                                    num_return_sequences=num_outputs,
                                    pad_token_id = 50256,
                                    )

    if num_outputs == 1:
        output = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        st.caption("Generated Quote")
        st.info(output + "")
    if num_outputs > 1:
        for (idx, item) in enumerate(sample_outputs):
            output = tokenizer.decode(item, skip_special_tokens=True)
            st.caption(f"Generated Quote #{idx + 1}")
            st.info(output + "")

    
    

if st.button('Generate'):
    with st.spinner("Please wait..."):
        run_model(generated_max_len, num_outputs)
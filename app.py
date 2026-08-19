import torch
from transformers import pipeline
import streamlit as st

# 1. Load the model efficiently so it doesn't crash on rerun
@st.cache_resource
def load_model():
  return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

pipe = load_model()

# 2. Initialize the session state memory
if "messages" not in st.session_state:
  st.session_state["messages"] = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a comedian",
    }
]

# 3. Display past chat history
for msg in st.session_state["messages"][1:]:
  with st.chat_message(msg["role"]):
    st.write(msg["content"])

# 4. Handle new user input
if prompt := st.chat_input("Type your message..."):
  # save and show the user's message
  st.session_state["messages"].append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.write(prompt)

  # generate the model's response
  model_prompt = pipe.tokenizer.apply_chat_template(st.session_state["messages"], tokenize=False, add_generation_prompt=True)
  outputs = pipe(model_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
  full_generated_text = outputs[0]["generated_text"]

  # Extract only the assistant's new response content.
  # The full_generated_text contains the prompt (which includes the system and user messages)
  # and then the model's generated response. We remove the prompt part to get only the new content.
  response = full_generated_text[len(model_prompt):].strip()

  # save and show the assistant's message
  st.session_state["messages"].append({"role": "assistant", "content": response})
  with st.chat_message("assistant"):
    st.write(response)

  if len(st.session_state["messages"]) > 11:
    st.session_state["messages"].pop(1)
    st.session_state["messages"].pop(1)

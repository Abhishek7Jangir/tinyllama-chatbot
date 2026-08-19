# Conversational Chatbot with TinyLlama-1.1B

A lightweight conversational chatbot built with TinyLlama-1.1B-Chat via Hugging Face's `pipeline` API, with conversational memory and a fixed-size context window. Built as a first hands-on project while working through Hugging Face's LLM course, and deployed live via Google Colab + Cloudflare Tunnel.

## What This Is (and Isn't)

This is a foundational project — a starting point for learning how LLM-based chatbots actually work under the hood, not a production system. I'm including it because the mechanics here (chat templates, conversation state, context windowing) are the same fundamentals that show up in more complex LLM applications, and I'd rather show honest, working basics than nothing at all.

## Features

- **Text generation** using TinyLlama-1.1B-Chat via Hugging Face's `pipeline()` function, with a configurable system persona (currently set to respond in a comedic style)
- **Conversational memory**: the full conversation (system prompt + all prior turns) is stored in Streamlit's session state and passed to the model on every turn, formatted using the model's own chat template (`apply_chat_template`) so the prompt structure matches what the model was trained on
- **Fixed-size context window**: once the conversation exceeds 11 stored messages (system prompt + 5 user/assistant exchanges), the oldest user/assistant pair is dropped while the system prompt is always kept, so the window stays bounded without losing the original persona instruction
- **Streamlit chat interface** with persistent message history displayed across turns
- **Public deployment via Cloudflare Tunnel**, run from Google Colab (`cloudflared tunnel`), to make the locally-running Streamlit app accessible over a public URL without separate hosting

## How It Works

1. User sends a message via the Streamlit chat interface
2. The message is appended to the session's message history
3. The full history is formatted into a single prompt string using the model's chat template, which correctly wraps each message with the role-specific formatting the model expects
4. The formatted prompt is passed to the TinyLlama pipeline for text generation (sampling with temperature 0.7, top-k 50, top-p 0.95)
5. The model's raw output includes the original prompt plus its new response — the prompt portion is sliced off to extract just the new reply
6. The response is displayed and appended to the session's message history
7. If the message history has grown beyond 11 entries, the oldest user/assistant pair is removed (the system prompt at the start is never removed) to keep the context window bounded

## Known Limitations (Honest Assessment)

- The context window is trimmed by **message count**, not actual token count — for very long individual messages, this could still exceed the model's real token limit. A token-aware trimming approach would be more robust.
- No summarization of dropped context — older turns are simply discarded rather than condensed, so long conversations lose earlier context entirely rather than retaining a compressed version of it.
- No retrieval or external knowledge — the model only knows what's in its training data and the current conversation.
- TinyLlama-1.1B is a small model — response quality and coherence are noticeably more limited than larger models.

## What's Next

Currently building a second, more advanced project: a RAG (Retrieval-Augmented Generation) system that grounds answers in actual document content, using proper embedding-based retrieval rather than relying purely on the model's training data and raw conversation memory.

## Setup & Installation

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

To expose it publicly (as run originally, from Google Colab):
```bash
wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb
streamlit run app.py & cloudflared tunnel --url http://localhost:8501
```

## Tech Stack

Python, Hugging Face Transformers (`pipeline`, `apply_chat_template`), TinyLlama-1.1B-Chat, PyTorch, Streamlit, Cloudflare Tunnel

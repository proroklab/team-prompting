"""API keys for the LLM/LMM data-generation and online-planner baseline.

Copy this file to ``api_keys.py`` (which is git-ignored) and fill in your own
keys, or export them as environment variables before running.

    cp api_keys.example.py api_keys.py

Only needed for:
  * Stage-1 dataset generation (Gemini), and
  * the online-LLM-planner baseline in evaluations/baselines (Groq / Llama 3.3 70B).
The released checkpoints and the RNN/GNN pipeline do **not** require any key.
"""
import os

# Google Gemini — https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Groq — https://console.groq.com/keys  (online-LLM-planner baseline)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

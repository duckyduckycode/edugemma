# EduGemma Setup Guide

## Prerequisites
1. Kaggle account with identity verification
2. Google AI Studio access (free tier)
3. Ollama installed locally (for demo)

## Step 1: Kaggle Setup
- Register at https://www.kaggle.com/
- Go to Settings → Identity Verification → verify with ID
- Accept competition rules at https://www.kaggle.com/competitions/gemma-4-good-hackathon
- Enable GPU notebooks (Settings → Accelerator)

## Step 2: Local Dev Environment
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma 4 E4B (for local testing)
ollama pull gemma4:e4b

# Pull 31B for quality (needs GPU or cloud)
# ollama pull gemma4:31b

# Python setup
pip install kaggle transformers torch unsloth
```

## Step 3: Fine-Tuning (on Kaggle GPU)
- Create Kaggle notebook with GPU T4
- Use Unsloth for efficient fine-tuning
- Training data: Khan Academy QA pairs, OpenStax textbooks

## Step 4: Build & Deploy
- Backend: Python FastAPI + Ollama
- Frontend: Svelte PWA (offline-capable)
- Demo: Screen recording of local usage

## Competition Timeline
- **Now → Apr 25:** Setup + data collection + initial prototype
- **Apr 26 → May 10:** Fine-tuning + feature development
- **May 11 → May 15:** Polish, video, documentation
- **May 16 → May 18:** Final submission

# Gemma 4 Good Hackathon — Action Plan

## Timeline
- **Now → Apr 25:** Ideation, research, Kaggle account setup, identity verification
- **Apr 26 → May 10:** Core development (2 weeks)
- **May 11 → May 15:** Polish, demo video, documentation
- **May 16 → May 18:** Final submission

## Hardware Constraints
- Fengzi's machine: Ryzen 5 3600, 16GB RAM, NO dedicated GPU
- Can't run Gemma 4 locally at speed
- **Solutions:**
  - Kaggle notebooks (free GPU — T4/P100)
  - Google AI Studio (free Gemma 4 API access)
  - Google Colab (free GPU)
  - Ollama for demo (CPU-only, slow but works for video)

## Top Project Pick: Offline Adaptive STEM Tutor

### Why This Project
1. **Education track** — less competitive than health
2. **Offline-first** — directly addresses hackathon's constrained-environment requirement
3. **Multimodal** — students can upload photos of problems (textbook, handwritten)
4. **Function calling** — adaptive difficulty, quiz generation, progress tracking
5. **Ollama deployment** — special track bonus
6. **Unsloth fine-tuning** — $10K separate prize for best fine-tuned model
7. **Strong demo** — screen recording of local tutor running is compelling
8. **Impact story** — rural classrooms, weak internet, underserved communities

### Project Description
**EduGemma: Offline AI Tutor for STEM Education**

An offline-first adaptive STEM tutor powered by fine-tuned Gemma 4. Students can:
- Ask questions in natural language (multilingual)
- Upload photos of textbook problems (multimodal vision)
- Get step-by-step explanations adapted to their level
- Take adaptive quizzes that adjust difficulty
- Track progress locally (privacy-first, no cloud)

Runs entirely on a laptop via Ollama. Fine-tuned on Khan Academy and OpenStax content using Unsloth.

### Tech Stack
- **Model:** Gemma 4 E4B (fine-tuned with Unsloth) for edge, 31B for server
- **Inference:** Ollama (local) + Kaggle GPU (fine-tuning)
- **Backend:** Python + FastAPI
- **Frontend:** Svelte/React PWA (works offline)
- **Fine-tuning dataset:** Khan Academy, OpenStax, custom STEM Q&A
- **Deployment:** Docker + Ollama

### Submission Requirements
1. ✅ Kaggle account + identity verification
2. ✅ Public project write-up
3. ✅ Public code repository (GitHub)
4. ✅ Public demo or demo files
5. ✅ Public video (3-5 min)
6. ✅ Cover image / media gallery

### Competition Analysis
- Most submissions will be generic chatbots or PDF assistants
- Standing out requires: specific problem, working offline demo, fine-tuned model, strong storytelling
- Our edge: actual offline deployment, fine-tuned model, multimodal, function calling

## Next Steps
1. [ ] Fengzi creates Kaggle account + verifies identity
2. [ ] Fengzi registers for the hackathon at https://www.kaggle.com/competitions/gemma-4-good-hackathon
3. [ ] Set up Kaggle notebook with GPU for fine-tuning
4. [ ] Start collecting training data (Khan Academy, OpenStax)
5. [ ] Build initial prototype with base Gemma 4 E4B via Ollama
6. [ ] Fine-tune on STEM education content with Unsloth

# Gemma 4 Hackathon — Technical Research

## Model Specs
| Model | Params | Active | Context | Modalities | Best For |
|-------|--------|--------|---------|------------|----------|
| E2B | 2.3B eff (5.1B total) | 2.3B | 128K | Text, Image, Audio | Mobile/edge |
| E4B | 4.5B eff (8B total) | 4.5B | 128K | Text, Image, Audio | Laptop/edge |
| 26B MoE | 26B total | 4B active | 128K | Text, Image | Workstation |
| 31B Dense | 30.7B | 30.7B | 256K | Text, Image | Server/workstation |

## Key Capabilities
- **Native function calling** — built-in tool use / agentic workflows
- **Multimodal** — text + image (all models), + audio (E2B/E4B only)
- **Configurable thinking modes** — reasoning capability
- **128K-256K context window** — huge
- **Apache 2.0 license** — unrestricted commercial use
- **Runs via Ollama** — `ollama run gemma4:e4b` etc.
- **Fine-tunable with Unsloth** — special $10K track

## Benchmarks (31B Dense)
- MMLU Pro: 85.2%
- AIME 2026: 89.2%
- LiveCodeBench v6: 80.0%
- Codeforces ELO: 2150
- GPQA Diamond: 84.3%
- MMMU Pro (vision): 76.9%

## Hackathon Tracks
1. **Health & Sciences** — clinical tools, medical literature, privacy-first
2. **Global Resilience** — disaster response, climate, offline coordination
3. **Future of Education** — adaptive learning, accessibility, multilingual

## Special Tracks
- **Ollama integration** — bonus for local deployment
- **Unsloth fine-tuning** — $10K separate prize

## What Judges Want
- Real-world impact (not generic demos)
- Works in constrained environments (low bandwidth, offline, privacy)
- Technical innovation (uses multimodal, function calling, NOT just chatbot)
- Practical deployment (demo must work)

## Winning Strategy
1. Pick a SPECIFIC problem (not "AI for education" but "offline STEM tutor for rural classrooms")
2. Use Gemma 4's unique features (multimodal, function calling, edge deployment)
3. Fine-tune with Unsloth for domain specificity (bonus $10K track)
4. Deploy via Ollama for offline/edge capability (special track bonus)
5. Make the demo impressive and the video compelling
6. Tell a strong impact story

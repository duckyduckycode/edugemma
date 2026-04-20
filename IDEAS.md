# Gemma 4 Good Hackathon — Project Ideas

## Criteria
- Must use Gemma 4 models (E2B, E4B, 26B MoE, 31B Dense)
- Must solve real problems in constrained environments (low-connectivity, privacy-critical, underserved)
- Must be technically credible and demonstrate multimodal + function calling
- Judged on: social impact, technical innovation, practical deployment

## Track: Future of Education
### Idea 1: Offline Adaptive Tutor for STEM
- Fine-tune Gemma 4 E4B on Khan Academy-style content
- Runs locally via Ollama on student laptops
- Adapts difficulty based on student responses
- Works offline in rural classrooms with weak internet
- Multilingual support (Chinese, Spanish, etc.)
- **Why it fits:** Directly addresses "classrooms with weak internet", uses Gemma 4's function calling for adaptive logic

### Idea 2: Code Literacy Agent for Beginners
- Gemma 4 26B MoE as a patient code tutor
- Explains code step-by-step with visual annotations
- Runs on low-end hardware (E2B/E4B variants)
- Supports multiple programming languages
- **Why it fits:** Makes coding education accessible without cloud access

## Track: Health & Sciences
### Idea 3: Local Clinical Intake Assistant
- Offline-first patient intake form + AI summarization
- Uses Gemma 4 E4B for local inference
- Converts patient descriptions into structured clinical notes
- Privacy-first: no data leaves the device
- Multilingual intake for immigrant communities
- **Why it fits:** Frontline healthcare, privacy-critical, low-connectivity

### Idea 4: Medical Literature Simplifier for Patients
- Takes complex lab results / medical documents
- Generates plain-language explanations using Gemma 4
- Works offline with local model
- **Why it fits:** Bridges gap between people and specialized knowledge

## Track: Global Resilience
### Idea 5: Field Incident Summarizer for Disaster Response
- First responders record voice/text incident reports
- Gemma 4 summarizes, categorizes, and prioritizes
- Works offline, syncs when connectivity available
- Multilingual for international disaster zones
- **Why it fits:** Practical resilience tool, edge AI, low-connectivity coordination

## Strongest Picks (ranked by win probability × impact)
1. **Idea 1 (Offline Adaptive Tutor)** — Education track is less saturated than health, strong storytelling angle, Ollama integration for special track bonus
2. **Idea 5 (Field Incident Summarizer)** — Climate/resilience track has special mentions, strong demo potential
3. **Idea 3 (Clinical Intake)** — Health track is competitive but impact story is strong

## Technical Stack
- Gemma 4 via Ollama (local inference)
- Unsloth for fine-tuning (special $10K track)
- Python + FastAPI backend
- Simple web frontend (React/Svelte)
- PWA for offline support

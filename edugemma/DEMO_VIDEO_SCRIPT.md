# EduGemma Demo Video Script
**Length:** 2-3 minutes (judges won't watch beyond 3 min)
**Tools needed:** Screen recorder (OBS, Win+G, or Loom), browser

## Setup (before recording)
1. Open `web/index.html` in Chrome/Edge
2. Start the backend: `cd api && pip install -r requirements.txt && uvicorn main:app --reload`
3. Make sure the status badge shows "Online" (green)
4. If backend won't start, demo mode (orange "Demo" badge) still works great

## Script

### 0:00 - Hook (15 seconds)
**Screen:** Browser with EduGemma open, zoomed out to show the full interface
**Narration:** "1.7 billion students can't access online AI tutors. EduGemma changes that — it's a fully offline STEM tutor that runs on any laptop, no internet required."

### 0:15 - The Problem (20 seconds)
**Screen:** Show the "Demo" badge, then type a question
**Narration:** "Most AI tutors — ChatGPT, Khanmigo — need constant internet. That's useless in rural schools, developing countries, or anywhere connectivity is unreliable. EduGemma works completely offline after a one-time download."

### 0:35 - Live Demo: Math (30 seconds)
**Screen:** Type "How do I find the derivative of x³ + 2x?"
**Narration:** "Let's try a calculus question. EduGemma gives step-by-step explanations at the student's level — not just the answer, but how to think about it."

### 1:05 - Adaptive Difficulty (30 seconds)
**Screen:** Click the quiz button, answer a few questions correctly, watch the difficulty increase
**Narration:** "The adaptive engine tracks mastery per topic. Get questions right, and it moves you up. Struggle, and it adjusts down. This isn't a static 'easy/medium/hard' label — it's dynamic per-topic tracking."

### 1:35 - Multilingual (20 seconds)
**Screen:** Type a question in Chinese: "解释牛顿第二定律"
**Narration:** "EduGemma responds in the student's language — not through UI translation, but because the model generates multilingual responses. A student in rural China learns physics in 中文."

### 1:55 - Privacy (15 seconds)
**Screen:** Show the browser dev tools → Application → SQLite database
**Narration:** "All data stays local. No accounts, no cloud, no analytics. Student records live in a local SQLite database. COPPA and FERPA compliance isn't a feature — it's the default."

### 2:10 - Technical Details (25 seconds)
**Screen:** Show the repo on GitHub, scroll through the training data
**Narration:** "Built on Gemma 4 E4B — the Mixture-of-Experts architecture makes it efficient enough for CPU inference. Fine-tuned with Unsloth on 500 curated STEM examples. The whole stack — PWA frontend, FastAPI backend, Ollama inference — installs in three commands."

### 2:35 - Impact & Closing (25 seconds)
**Screen:** Show the cover image / architecture diagram
**Narration:** "One school lab installation serves hundreds of students at zero recurring cost. EduGemma isn't another cloud chatbot — it's education infrastructure for the 1.7 billion students the internet left behind. Thank you."

## Recording Tips
- Use Chrome in dark mode for best visual impact
- Keep the window size consistent (1280x720 or 1920x1080)
- Speak clearly and at a moderate pace
- Don't edit too much — authentic is better than polished
- Upload to YouTube as "Public" or "Unlisted"

## After Recording
- Upload to YouTube
- Add the link to SUBMISSION.md
- Update README.md with the video link

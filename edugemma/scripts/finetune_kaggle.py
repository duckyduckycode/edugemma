"""
EduGemma — Kaggle Fine-Tuning Notebook
Fine-tune Gemma 4 E4B on STEM education data using Unsloth

Run this on Kaggle with GPU T4 or P100 enabled.
Based on Unsloth's Gemma 4 fine-tuning guide.
"""

# === Kaggle Notebook Setup ===
# 1. Create new Kaggle notebook
# 2. Settings → Accelerator → GPU T4
# 3. Settings → Internet → On (for pip install)
# 4. Upload training data as dataset

# %% [code]
# Install Unsloth and dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

# %% [code]
from unsloth import FastModel
import torch

# Load Gemma 4 E4B with 4-bit quantization (fits on T4)
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-e4b",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# %% [code]
# Add LoRA adapters for efficient fine-tuning
model = FastModel.get_peft_model(
    model,
    r=16,           # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# %% [code]
# Load training data
from datasets import load_dataset

# Option 1: Load from Kaggle dataset
# dataset = load_dataset("json", data_files="/kaggle/input/edugemma-training/training_data.jsonl")

# Option 2: Load from HuggingFace
# dataset = load_dataset("your-username/edugemma-stem-qa")

# For now, use the seed data
import json
train_data = []
with open("data/training/unsloth_training_data.jsonl") as f:
    for line in f:
        train_data.append(json.loads(line))

print(f"Training examples: {len(train_data)}")
print(f"First example preview: {train_data[0]['text'][:200]}...")

# %% [code]
# Set up trainer
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="outputs",
    ),
)

# %% [code]
# Train!
trainer.train()

# %% [code]
# Save the model
model.save_pretrained("edugemma-e4b-lora")
tokenizer.save_pretrained("edugemma-e4b-lora")

# Optional: Save merged model for Ollama
model.save_pretrained_merged("edugemma-e4b-merged", tokenizer)

# %% [code]
# Test the fine-tuned model
messages = [
    {"role": "system", "content": "You are EduGemma, an adaptive STEM tutor."},
    {"role": "user", "content": "How do I find the derivative of x^3?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    use_cache=True,
)

response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(f"EduGemma: {response}")

# %% [code]
# Export to GGUF for Ollama
model.save_pretrained_gguf("edugemma-e4b", tokenizer, quantization_method="q4_k_m")

# Upload to Kaggle as output for download
print("Fine-tuned model saved! Download edugemma-e4b-q4_k_m.gguf for Ollama deployment.")

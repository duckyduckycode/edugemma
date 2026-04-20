"""
EduGemma - Training Data Collector
Collects and formats STEM education Q&A pairs for fine-tuning Gemma 4 E4B
"""
import json
import os

OUTPUT_DIR = "data/training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAINING_DATA = [
    {
        "conversations": [
            {"role": "user", "content": "What is 2x + 3 = 7? I don't know where to start."},
            {"role": "assistant", "content": "Let's work through this step by step!\n\nWe have: 2x + 3 = 7\n\nStep 1: We want to get x by itself. First, let's deal with the +3.\nSubtract 3 from both sides:\n2x + 3 - 3 = 7 - 3\n2x = 4\n\nStep 2: Now divide both sides by 2:\n2x / 2 = 4 / 2\nx = 2\n\nLet's check: 2(2) + 3 = 4 + 3 = 7\n\nThe answer is x = 2!"}
        ],
        "metadata": {"topic": "algebra", "difficulty": 1, "source": "curated"}
    },
    {
        "conversations": [
            {"role": "user", "content": "How do I solve 3(x - 2) = 15?"},
            {"role": "assistant", "content": "Great question! There are two ways to solve this.\n\nMethod 1: Distribute first\n3(x - 2) = 15\n3x - 6 = 15\n3x = 21\nx = 7\n\nMethod 2: Divide first\n3(x - 2) = 15\n(x - 2) = 5\nx = 7\n\nBoth give us x = 7! Method 2 is often faster when the number outside the parentheses divides evenly.\n\nCheck: 3(7 - 2) = 3(5) = 15"}
        ],
        "metadata": {"topic": "algebra", "difficulty": 2, "source": "curated"}
    },
    {
        "conversations": [
            {"role": "user", "content": "I keep confusing the product rule and chain rule. How do I know which one to use?"},
            {"role": "assistant", "content": "This is super common! Here's how I think about it:\n\nProduct Rule: when you're multiplying TWO functions\nf(x) = x^2 * sin(x)  ->  two things multiplied together\n\nChain Rule: when one function is INSIDE another\nf(x) = sin(x^2)  ->  sin of something, not sin times something\n\nQuick test: Can you put a multiplication sign between the parts?\n- x^2 * sin(x) -> YES -> Product rule\n- sin(x^2) -> NO (it's sin OF x^2) -> Chain rule\n\nTricky case: What about x^2 * sin(x^3)?\nYou need BOTH! Product rule on the outside (x^2 times sin(...)), then chain rule on sin(x^3)."}
        ],
        "metadata": {"topic": "calculus", "difficulty": 3, "source": "curated"}
    },
    {
        "conversations": [
            {"role": "user", "content": "A ball is thrown up at 20 m/s. When does it come back down?"},
            {"role": "assistant", "content": "Let's solve this using kinematic equations.\n\nGiven: v0 = 20 m/s (upward), g = 9.8 m/s^2 (downward)\n\nAt the highest point, velocity = 0.\n\nv = v0 - gt\n0 = 20 - 9.8t\nt = 20/9.8 = 2.04 seconds\n\nBut that's just going UP! The ball takes the same time coming down.\n\nTotal time = 2 * 2.04 = 4.08 seconds\n\nAlternative: Use y = v0t - (1/2)gt^2 and find when y = 0:\n0 = 20t - 4.9t^2\nt(20 - 4.9t) = 0\nt = 0 or t = 20/4.9 = 4.08 s"}
        ],
        "metadata": {"topic": "physics_kinematics", "difficulty": 2, "source": "curated"}
    },
    {
        "conversations": [
            {"role": "user", "content": "What's the difference between an ionic bond and a covalent bond?"},
            {"role": "assistant", "content": "Think of it like sharing vs giving away:\n\nIonic Bond = 'I give, you take'\n- One atom gives electrons to another\n- Usually between a metal and nonmetal\n- Example: NaCl (sodium gives 1 electron to chlorine)\n- Creates charged particles (ions) that stick together\n\nCovalent Bond = 'Let's share!'\n- Atoms share electrons between them\n- Usually between two nonmetals\n- Example: H2O (oxygen shares electrons with 2 hydrogens)\n- Creates molecules\n\nMemory trick:\n- IONic = I Owe N (electron) - one gives it away\n- CovaLENT = sharing is elegant"}
        ],
        "metadata": {"topic": "chemistry_bonding", "difficulty": 1, "source": "curated"}
    },
]

def convert_to_unsloth_format(data):
    """Convert to Unsloth's chatml format for fine-tuning"""
    converted = []
    for entry in data:
        messages = entry["conversations"]
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        converted.append({"text": text})
    return converted


def main():
    with open(f"{OUTPUT_DIR}/raw_training_data.json", "w") as f:
        json.dump(TRAINING_DATA, f, indent=2)
    print(f"Saved {len(TRAINING_DATA)} raw training examples")

    unsloth_data = convert_to_unsloth_format(TRAINING_DATA)
    with open(f"{OUTPUT_DIR}/unsloth_training_data.jsonl", "w") as f:
        for entry in unsloth_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(unsloth_data)} Unsloth-format examples")

    topics = set()
    difficulties = {}
    for entry in TRAINING_DATA:
        meta = entry["metadata"]
        topics.add(meta["topic"])
        difficulties[meta["difficulty"]] = difficulties.get(meta["difficulty"], 0) + 1

    stats = {
        "total_examples": len(TRAINING_DATA),
        "topics": list(topics),
        "difficulty_distribution": difficulties,
        "note": "This is seed data. Need to expand to 500+ examples for fine-tuning."
    }
    with open(f"{OUTPUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nTopics: {', '.join(topics)}")
    print(f"Difficulty distribution: {difficulties}")
    print(f"\nNeed 500+ examples for effective fine-tuning. Current: {len(TRAINING_DATA)}")


if __name__ == "__main__":
    main()

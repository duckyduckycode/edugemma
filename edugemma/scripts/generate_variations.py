"""
EduGemma - Programmatic Training Data Generator
Generates variations of problems to bulk up the dataset to 500+
"""
import json
import random
import os

OUTPUT_DIR = "data/training"

def gen_algebra_variations():
    """Generate algebra problems with varying numbers"""
    entries = []
    
    # Linear equations: ax + b = c (difficulty 1-2)
    for a in [2, 3, 4, 5, 6, 7, 8]:
        for b in [1, 3, 5, 7, -2, -4, -6]:
            c = a * random.randint(2, 10) + b
            if c <= 0: continue
            x = (c - b) / a
            if x != int(x): continue
            x = int(x)
            sign = "+" if b >= 0 else "-"
            abs_b = abs(b)
            q = f"Solve {a}x {sign} {abs_b} = {c}"
            a_step = f"{a}x {sign} {abs_b} = {c}\n"
            if b > 0:
                a_step += f"Subtract {abs_b} from both sides:\n{a}x = {c - b}\n"
            else:
                a_step += f"Add {abs_b} to both sides:\n{a}x = {c - b}\n"
            a_step += f"Divide by {a}:\nx = {x}\n\nCheck: {a}({x}) {sign} {abs_b} = {a*x + b}"
            entries.append((q, a_step, "algebra", 1 if a <= 3 else 2))
    
    # Two-step: ax + b = cx + d (difficulty 2)
    for _ in range(30):
        a = random.randint(2, 6)
        b = random.randint(-10, 10)
        c = random.randint(1, a-1) if a > 1 else 1
        d = random.randint(-10, 10)
        x_num = d - b
        x_den = a - c
        if x_den == 0 or x_num % x_den != 0: continue
        x = x_num // x_den
        if abs(x) > 20: continue
        sign_b = "+" if b >= 0 else "-"
        sign_d = "+" if d >= 0 else "-"
        q = f"Solve {a}x {sign_b} {abs(b)} = {c}x {sign_d} {abs(d)}"
        a_text = f"{a}x {sign_b} {abs(b)} = {c}x {sign_d} {abs(d)}\n"
        a_text += f"Move x terms to one side:\n{a-c}x = {d-b}\n"
        a_text += f"x = {x}\n\nCheck: {a}({x}){'+' if b>=0 else ''}{b} = {a*x+b}, {c}({x}){'+' if d>=0 else ''}{d} = {c*x+d}"
        entries.append((q, a_text, "algebra", 2))
    
    return entries[:80]  # Cap at 80

def gen_calculus_variations():
    """Generate calculus differentiation/integration problems"""
    entries = []
    
    # Power rule differentiation (difficulty 1-2)
    powers = [2, 3, 4, 5, -1, -2, 0.5, 1.5, -0.5]
    for n in powers:
        coeffs = [1, 2, 3, 4, 5, -1, -2, -3]
        for c in coeffs[:4]:
            if n == 2:
                q = f"Find the derivative of {c}x²"
            elif n == 0.5:
                q = f"Find the derivative of {c}√x" if c > 0 else f"Find the derivative of -{abs(c)}√x"
            elif n == -1:
                q = f"Find the derivative of {c}/x" if c > 0 else f"Find the derivative of {abs(c)}/x"
            elif n == -2:
                q = f"Find the derivative of {c}/x²" if c > 0 else f"Find the derivative of -{abs(c)}/x²"
            else:
                q = f"Find the derivative of {c}x^{n}"
            
            # d/dx(cx^n) = cn*x^(n-1)
            new_c = c * n
            new_n = n - 1
            if new_n == 0:
                ans = f"Using the power rule: d/dx({c}x^{n}) = {c}·{n}·x^{n-1} = {new_c}"
            elif new_n == 1:
                ans = f"Using the power rule: d/dx({c}x^{n}) = {c}·{n}·x^{n-1} = {new_c}x"
            else:
                ans = f"Using the power rule: d/dx({c}x^{n}) = {c}·{n}·x^{n-1} = {new_c}x^{new_n}"
            entries.append((q, ans, "calculus", 1 if n in [2,3,4] else 2))
    
    # Integration power rule (difficulty 2)
    for n in [1, 2, 3, 4, -2]:
        for c in [1, 2, 3, 5]:
            if n == -1:
                continue  # 1/x integration is ln|x|
            new_n = n + 1
            new_c_str = f"{c}/{new_n}" if c % new_n != 0 else f"{c // new_n}"
            q = f"Find ∫ {c}x^{n} dx"
            ans = f"Reverse power rule: ∫ {c}x^{n} dx = {c}·x^{new_n}/{new_n} + C = {new_c_str}x^{new_n} + C\n\nCheck: d/dx({new_c_str}x^{new_n} + C) = {c}x^{n}"
            entries.append((q, ans, "calculus", 2))
    
    return entries[:60]

def gen_physics_variations():
    """Generate physics calculation problems"""
    entries = []
    
    # F = ma problems (difficulty 2)
    for _ in range(15):
        m = random.choice([2, 3, 5, 8, 10, 15, 20, 25])
        a = random.choice([1, 2, 3, 4, 5, 6, 8, 10])
        F = m * a
        q = f"A {m} kg object accelerates at {a} m/s². What force is needed?"
        ans = f"Newton's Second Law: F = ma\n\nF = {m} × {a} = {F} N\n\nA force of {F} Newtons is needed."
        entries.append((q, ans, "physics", 2))
    
    # Kinetic energy problems (difficulty 2)
    for _ in range(10):
        m = random.choice([1, 2, 3, 5, 8, 10])
        v = random.choice([2, 3, 4, 5, 6, 8, 10, 15, 20])
        ke = 0.5 * m * v**2
        q = f"What is the kinetic energy of a {m} kg object moving at {v} m/s?"
        ans = f"KE = ½mv²\nKE = ½ × {m} × {v}²\nKE = ½ × {m} × {v**2}\nKE = {ke} J\n\nThe kinetic energy is {ke} Joules."
        entries.append((q, ans, "physics", 2))
    
    # Potential energy problems (difficulty 2)
    for _ in range(10):
        m = random.choice([1, 2, 5, 10])
        h = random.choice([2, 3, 5, 8, 10, 15, 20])
        g = 9.8
        pe = m * g * h
        q = f"What is the gravitational potential energy of a {m} kg object at height {h} m?"
        ans = f"PE = mgh\nPE = {m} × 9.8 × {h}\nPE = {pe:.1f} J\n\nThe potential energy is approximately {pe:.1f} Joules."
        entries.append((q, ans, "physics", 2))
    
    # Ohm's Law problems (difficulty 2)
    for _ in range(10):
        V = random.choice([3, 5, 6, 9, 12, 24, 120])
        R = random.choice([10, 20, 50, 100, 200, 500, 1000])
        I = V / R
        q = f"A {V}V battery is connected to a {R}Ω resistor. What current flows?"
        if I >= 0.001:
            ans = f"Ohm's Law: V = IR\nI = V/R = {V}/{R} = {I:.4f} A = {I*1000:.1f} mA"
        else:
            ans = f"Ohm's Law: V = IR\nI = V/R = {V}/{R} = {I:.6f} A = {I*1000000:.1f} μA"
        entries.append((q, ans, "physics", 2))
    
    return entries[:45]

def gen_chemistry_variations():
    """Generate chemistry calculation problems"""
    entries = []
    
    # Molar mass calculations (difficulty 2)
    compounds = [
        ("NaCl", [("Na", 22.99, 1), ("Cl", 35.45, 1)]),
        ("H₂O", [("H", 1.008, 2), ("O", 16.00, 1)]),
        ("CO₂", [("C", 12.01, 1), ("O", 16.00, 2)]),
        ("C₆H₁₂O₆", [("C", 12.01, 6), ("H", 1.008, 12), ("O", 16.00, 6)]),
        ("Ca(OH)₂", [("Ca", 40.08, 1), ("O", 16.00, 2), ("H", 1.008, 2)]),
        ("Fe₂O₃", [("Fe", 55.85, 2), ("O", 16.00, 3)]),
        ("NH₃", [("N", 14.01, 1), ("H", 1.008, 3)]),
        ("CH₄", [("C", 12.01, 1), ("H", 1.008, 4)]),
        ("H₂SO₄", [("H", 1.008, 2), ("S", 32.06, 1), ("O", 16.00, 4)]),
        ("NaOH", [("Na", 22.99, 1), ("O", 16.00, 1), ("H", 1.008, 1)]),
    ]
    for name, elements in compounds:
        total = sum(aw * count for _, aw, count in elements)
        breakdown = " + ".join([f"{count}×{aw}" if count > 1 else f"{aw}" for _, aw, count in elements])
        q = f"Calculate the molar mass of {name}"
        ans = f"{name}:\n" + "\n".join([f"{el}: {count} × {aw} = {count*aw:.3f}" for el, aw, count in elements])
        ans += f"\n\nTotal = {total:.2f} g/mol"
        entries.append((q, ans, "chemistry", 2))
    
    # Moles to grams / grams to moles (difficulty 2)
    for name, elements in compounds[:6]:
        mm = sum(aw * count for _, aw, count in elements)
        grams = random.choice([10, 25, 50, 100, 250])
        moles = grams / mm
        q = f"How many moles are in {grams} g of {name}? (Molar mass = {mm:.2f} g/mol)"
        ans = f"n = mass / molar mass\nn = {grams} / {mm:.2f}\nn = {moles:.3f} mol\n\nThere are {moles:.3f} moles in {grams} g of {name}."
        entries.append((q, ans, "chemistry", 2))
    
    return entries[:25]

def gen_geometry_variations():
    """Generate geometry calculation problems"""
    entries = []
    
    # Pythagorean theorem (difficulty 1-2)
    triples = [(3,4,5), (5,12,13), (8,15,17), (7,24,25), (6,8,10), (9,12,15)]
    for a, b, c in triples:
        q = f"A right triangle has legs of length {a} and {b}. Find the hypotenuse."
        ans = f"Pythagorean theorem: a² + b² = c²\n{a}² + {b}² = c²\n{a**2} + {b**2} = c²\nc² = {a**2 + b**2}\nc = {c}"
        entries.append((q, ans, "geometry", 1))
    
    # Find missing leg
    for a, b, c in triples[:4]:
        q = f"A right triangle has a hypotenuse of {c} and one leg of {a}. Find the other leg."
        ans = f"c² - a² = b²\n{c}² - {a}² = b²\n{c**2} - {a**2} = b²\nb² = {c**2 - a**2}\nb = {b}"
        entries.append((q, ans, "geometry", 2))
    
    # Area of shapes (difficulty 1-2)
    for r in [3, 5, 7, 10, 12]:
        q = f"Find the area of a circle with radius {r}"
        ans = f"A = πr²\nA = π({r})² = {r**2}π ≈ {3.14159 * r**2:.2f}"
        entries.append((q, ans, "geometry", 1))
    
    for l, w in [(4,6), (5,8), (7,3), (10,15)]:
        q = f"Find the area and perimeter of a rectangle with length {l} and width {w}"
        ans = f"Area = length × width = {l} × {w} = {l*w}\nPerimeter = 2(length + width) = 2({l} + {w}) = {2*(l+w)}"
        entries.append((q, ans, "geometry", 1))
    
    # Volume (difficulty 2)
    for r, h in [(3,10), (5,8), (4,12)]:
        q = f"Find the volume of a cylinder with radius {r} and height {h}"
        ans = f"V = πr²h\nV = π({r})²({h})\nV = π({r**2})({h})\nV = {r**2 * h}π ≈ {3.14159 * r**2 * h:.2f}"
        entries.append((q, ans, "geometry", 2))
    
    for r in [2, 3, 5, 7]:
        q = f"Find the volume of a sphere with radius {r}"
        ans = f"V = (4/3)πr³\nV = (4/3)π({r})³\nV = (4/3)π({r**3})\nV = {4*r**3/3:.2f}π ≈ {4/3 * 3.14159 * r**3:.2f}"
        entries.append((q, ans, "geometry", 2))
    
    return entries[:40]

def gen_trig_variations():
    """Generate trig problems"""
    entries = []
    
    # Unit circle values (difficulty 2-3)
    angles = [(0, "0"), (30, "π/6"), (45, "π/4"), (60, "π/3"), (90, "π/2"),
              (120, "2π/3"), (135, "3π/4"), (150, "5π/6"), (180, "π")]
    
    for deg, rad in angles[:5]:
        import math
        s = round(math.sin(math.radians(deg)), 4)
        c = round(math.cos(math.radians(deg)), 4)
        if abs(s) < 0.0001: s = 0
        if abs(c) < 0.0001: c = 0
        # Clean up for display
        s_disp = {0: "0", 0.5: "1/2", 0.7071: "√2/2", 0.866: "√3/2", 1: "1"}.get(s, str(s))
        c_disp = {0: "0", 0.5: "1/2", 0.7071: "√2/2", 0.866: "√3/2", 1: "1"}.get(c, str(c))
        q = f"Find sin({deg}°) and cos({deg}°)"
        ans = f"Using the unit circle or special triangles:\nsin({deg}°) = {s_disp}\ncos({deg}°) = {c_disp}"
        entries.append((q, ans, "geometry", 2))
    
    # Inverse trig (difficulty 3)
    problems = [
        ("sin(θ) = 0.5, 0° ≤ θ ≤ 90°", "θ = 30° (or π/6 radians)\nsin(30°) = 1/2 = 0.5"),
        ("cos(θ) = √2/2, 0° ≤ θ ≤ 90°", "θ = 45° (or π/4 radians)\ncos(45°) = √2/2"),
        ("tan(θ) = √3, 0° ≤ θ ≤ 90°", "θ = 60° (or π/3 radians)\ntan(60°) = √3/1 = √3"),
        ("sin(θ) = √3/2, 0° ≤ θ ≤ 90°", "θ = 60° (or π/3 radians)\nsin(60°) = √3/2"),
        ("cos(θ) = 0.5, 0° ≤ θ ≤ 90°", "θ = 60° (or π/3 radians)\ncos(60°) = 1/2 = 0.5"),
    ]
    for q, a in problems:
        entries.append((f"Find θ if {q}", a, "geometry", 3))
    
    return entries[:15]

def generate_all():
    random.seed(42)
    entries = []
    entries.extend(gen_algebra_variations())
    entries.extend(gen_calculus_variations())
    entries.extend(gen_physics_variations())
    entries.extend(gen_chemistry_variations())
    entries.extend(gen_geometry_variations())
    entries.extend(gen_trig_variations())
    
    random.shuffle(entries)
    print(f"Programmatically generated {len(entries)} examples")
    
    # Load existing and merge
    existing = []
    with open(f"{OUTPUT_DIR}/raw_training_data.json") as f:
        existing = json.load(f)
    
    # Convert new entries to raw format
    new_raw = []
    for q, a, topic, diff in entries:
        new_raw.append({
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ],
            "metadata": {"topic": topic, "difficulty": diff, "source": "generated"}
        })
    
    combined = existing + new_raw
    
    # Save combined
    with open(f"{OUTPUT_DIR}/raw_training_data.json", "w") as f:
        json.dump(combined, f, indent=2)
    
    # Convert all to unsloth format
    unsloth = []
    for entry in combined:
        text = ""
        for msg in entry["conversations"]:
            text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        unsloth.append({"text": text})
    
    random.shuffle(unsloth)
    with open(f"{OUTPUT_DIR}/unsloth_training_data.jsonl", "w") as f:
        for entry in unsloth:
            f.write(json.dumps(entry) + "\n")
    
    # Stats
    topics = {}
    diffs = {}
    for entry in combined:
        t = entry["metadata"]["topic"]
        d = entry["metadata"]["difficulty"]
        topics[t] = topics.get(t, 0) + 1
        diffs[d] = diffs.get(d, 0) + 1
    
    stats = {
        "total_examples": len(combined),
        "topics": topics,
        "difficulty_distribution": diffs,
        "sources": {"curated": len(existing), "generated": len(new_raw)},
        "target": "500+ for effective fine-tuning"
    }
    with open(f"{OUTPUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTotal dataset: {len(combined)} examples")
    print(f"Topics: {json.dumps(topics, indent=2)}")
    print(f"Difficulty: {json.dumps(diffs, indent=2)}")

if __name__ == "__main__":
    generate_all()

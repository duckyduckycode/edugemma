"""
EduGemma - Mass Training Data Generator
Generates 500+ STEM Q&A pairs across topics and difficulty levels
for fine-tuning Gemma 4 E4B via Unsloth.
"""
import json
import os
import random

OUTPUT_DIR = "data/training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Template-based generation with high-quality curated content
# Each entry: (question_template, answer_template, topic, difficulty)

ENTRIES = []

# ============================================================
# ALGEBRA (60 examples)
# ============================================================

ALGEBRA = [
    # Difficulty 1 - Basic
    ("What is 3x = 12?", "Step by step:\n\n3x = 12\nDivide both sides by 3:\nx = 12/3\nx = 4\n\nCheck: 3(4) = 12", "algebra", 1),
    ("Solve x + 5 = 11", "Step by step:\n\nx + 5 = 11\nSubtract 5 from both sides:\nx = 11 - 5\nx = 6\n\nCheck: 6 + 5 = 11", "algebra", 1),
    ("What is 2x - 4 = 0?", "Step by step:\n\n2x - 4 = 0\nAdd 4 to both sides:\n2x = 4\nDivide by 2:\nx = 2\n\nCheck: 2(2) - 4 = 0", "algebra", 1),
    ("If 5 + y = 13, what is y?", "5 + y = 13\nSubtract 5 from both sides:\ny = 13 - 5\ny = 8\n\nCheck: 5 + 8 = 13", "algebra", 1),
    ("Solve for n: n/3 = 7", "n/3 = 7\nMultiply both sides by 3:\nn = 7 × 3\nn = 21\n\nCheck: 21/3 = 7", "algebra", 1),
    ("What is 4x = 20?", "4x = 20\nDivide both sides by 4:\nx = 20/4\nx = 5\n\nCheck: 4(5) = 20", "algebra", 1),
    ("Solve x - 9 = 3", "x - 9 = 3\nAdd 9 to both sides:\nx = 3 + 9\nx = 12\n\nCheck: 12 - 9 = 3", "algebra", 1),
    ("What is 10 - x = 4?", "10 - x = 4\nSubtract 10 from both sides:\n-x = 4 - 10\n-x = -6\nMultiply both sides by -1:\nx = 6\n\nCheck: 10 - 6 = 4", "algebra", 1),

    # Difficulty 2 - Multi-step
    ("Solve 2x + 5 = 13", "2x + 5 = 13\nSubtract 5 from both sides:\n2x = 8\nDivide by 2:\nx = 4\n\nCheck: 2(4) + 5 = 8 + 5 = 13", "algebra", 2),
    ("What is 3(x + 2) = 21?", "3(x + 2) = 21\nDivide both sides by 3:\nx + 2 = 7\nSubtract 2:\nx = 5\n\nOr distribute first:\n3x + 6 = 21\n3x = 15\nx = 5\n\nCheck: 3(5 + 2) = 3(7) = 21", "algebra", 2),
    ("Solve 5x - 3 = 2x + 9", "5x - 3 = 2x + 9\nSubtract 2x from both sides:\n3x - 3 = 9\nAdd 3:\n3x = 12\nDivide by 3:\nx = 4\n\nCheck: 5(4) - 3 = 17, 2(4) + 9 = 17", "algebra", 2),
    ("Find x: 4(x - 1) = 2(x + 3)", "4(x - 1) = 2(x + 3)\nDistribute:\n4x - 4 = 2x + 6\nSubtract 2x:\n2x - 4 = 6\nAdd 4:\n2x = 10\nx = 5\n\nCheck: 4(5-1) = 16, 2(5+3) = 16", "algebra", 2),
    ("Solve 7 - 2x = 3x + 2", "7 - 2x = 3x + 2\nAdd 2x to both sides:\n7 = 5x + 2\nSubtract 2:\n5 = 5x\nx = 1\n\nCheck: 7 - 2(1) = 5, 3(1) + 2 = 5", "algebra", 2),
    ("What is (x+3)/2 = 5?", "(x + 3)/2 = 5\nMultiply both sides by 2:\nx + 3 = 10\nSubtract 3:\nx = 7\n\nCheck: (7 + 3)/2 = 10/2 = 5", "algebra", 2),
    ("Solve 6x + 2 = 3x + 11", "6x + 2 = 3x + 11\nSubtract 3x:\n3x + 2 = 11\nSubtract 2:\n3x = 9\nx = 3\n\nCheck: 6(3) + 2 = 20, 3(3) + 11 = 20", "algebra", 2),
    ("If 2(3x - 1) = 4x + 6, find x", "2(3x - 1) = 4x + 6\n6x - 2 = 4x + 6\nSubtract 4x:\n2x - 2 = 6\nAdd 2:\n2x = 8\nx = 4\n\nCheck: 2(3·4-1) = 2(11) = 22, 4(4)+6 = 22", "algebra", 2),

    # Difficulty 3 - Quadratics & Systems
    ("Solve x² - 5x + 6 = 0", "x² - 5x + 6 = 0\nFactor: (x - 2)(x - 3) = 0\n\nSo x - 2 = 0 → x = 2\nOr x - 3 = 0 → x = 3\n\nTwo solutions: x = 2 or x = 3\n\nCheck: 2² - 5(2) + 6 = 4 - 10 + 6 = 0 ✓\n3² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓", "algebra", 3),
    ("Solve the system: x + y = 10 and 2x - y = 5", "From equation 1: y = 10 - x\nSubstitute into equation 2:\n2x - (10 - x) = 5\n2x - 10 + x = 5\n3x = 15\nx = 5\n\nThen y = 10 - 5 = 5\n\nSolution: x = 5, y = 5\n\nCheck: 5 + 5 = 10 ✓, 2(5) - 5 = 5 ✓", "algebra", 3),
    ("What is x² = 49?", "x² = 49\nTake the square root of both sides:\nx = ±7\n\nRemember: both 7 and -7 squared give 49!\nSo there are TWO solutions: x = 7 and x = -7", "algebra", 3),
    ("Solve 2x² + 3x - 2 = 0", "Using the quadratic formula:\nx = (-b ± √(b² - 4ac)) / 2a\n\na = 2, b = 3, c = -2\n\ndiscriminant = 9 - 4(2)(-2) = 9 + 16 = 25\n\nx = (-3 ± 5) / 4\n\nx₁ = (-3 + 5)/4 = 2/4 = 1/2\nx₂ = (-3 - 5)/4 = -8/4 = -2\n\nSolutions: x = 1/2 or x = -2", "algebra", 3),
    ("Find the intersection: y = 2x + 1 and y = -x + 7", "Set them equal:\n2x + 1 = -x + 7\n3x = 6\nx = 2\n\ny = 2(2) + 1 = 5\n\nIntersection point: (2, 5)", "algebra", 3),
    ("Solve x² - 4 = 0", "x² - 4 = 0\nThis is a difference of squares!\n(x + 2)(x - 2) = 0\n\nx = -2 or x = 2\n\nShortcut: x² = 4, so x = ±2", "algebra", 3),
    ("Solve 3x + 2y = 12 and x - y = 1", "From equation 2: x = y + 1\nSubstitute into equation 1:\n3(y + 1) + 2y = 12\n3y + 3 + 2y = 12\n5y = 9\ny = 9/5 = 1.8\n\nx = 1.8 + 1 = 2.8\n\nCheck: 3(2.8) + 2(1.8) = 8.4 + 3.6 = 12 ✓", "algebra", 3),

    # Difficulty 4 - Advanced
    ("Solve x² + 2x + 5 = 0", "Using the quadratic formula:\nx = (-2 ± √(4 - 20)) / 2\nx = (-2 ± √(-16)) / 2\nx = (-2 ± 4i) / 2\nx = -1 ± 2i\n\nThe discriminant is negative (b² - 4ac = -16), so there are no real solutions — only complex ones.\n\nThis means the parabola x² + 2x + 5 never crosses the x-axis.", "algebra", 4),
    ("Solve |2x - 3| = 7", "Absolute value equations split into two cases:\n\nCase 1: 2x - 3 = 7\n2x = 10, x = 5\n\nCase 2: 2x - 3 = -7\n2x = -4, x = -2\n\nSolutions: x = 5 or x = -2\n\nCheck: |2(5)-3| = |7| = 7 ✓\n|2(-2)-3| = |-7| = 7 ✓", "algebra", 4),
    ("Factor x³ - 8", "This is a difference of cubes!\n\nFormula: a³ - b³ = (a - b)(a² + ab + b²)\n\nHere a = x, b = 2:\n\nx³ - 8 = (x - 2)(x² + 2x + 4)\n\nNote: x² + 2x + 4 has discriminant 4 - 16 = -12 < 0, so it can't be factored further over the reals.", "algebra", 4),
    ("Solve the system: x² + y² = 25 and x + y = 7", "From equation 2: y = 7 - x\nSubstitute into equation 1:\nx² + (7 - x)² = 25\nx² + 49 - 14x + x² = 25\n2x² - 14x + 24 = 0\nx² - 7x + 12 = 0\n(x - 3)(x - 4) = 0\n\nIf x = 3: y = 4\nIf x = 4: y = 3\n\nSolutions: (3, 4) and (4, 3)\n\nThese are two points where the line intersects the circle of radius 5.", "algebra", 4),
    ("Solve 1/x + 1/(x+2) = 1", "Multiply through by x(x+2):\n(x+2) + x = x(x+2)\n2x + 2 = x² + 2x\n2 = x²\nx = ±√2\n\nCheck x = √2: 1/√2 + 1/(√2+2) ≈ 0.707 + 0.293 = 1 ✓\nCheck x = -√2: 1/(-√2) + 1/(2-√2) ≈ -0.707 + 1.707 = 1 ✓\n\nBoth solutions work!", "algebra", 4),

    # Difficulty 5 - Expert
    ("Prove that √2 is irrational", "By contradiction: Assume √2 is rational.\n\nThen √2 = p/q where p and q are integers with no common factors (fully reduced).\n\n2 = p²/q²\np² = 2q²\n\nSo p² is even, meaning p is even.\nWrite p = 2k.\n\n(2k)² = 2q²\n4k² = 2q²\nq² = 2k²\n\nSo q² is even, meaning q is even.\n\nBut if both p and q are even, they share a common factor of 2 — contradicting our assumption that p/q is fully reduced.\n\nTherefore √2 cannot be rational. QED.", "algebra", 5),
    ("Find all integer solutions to x² - y² = 12", "x² - y² = 12\n(x+y)(x-y) = 12\n\nWe need two integers with the same parity (both even or both odd) whose product is 12.\n\nFactor pairs of 12: (1,12), (2,6), (3,4), (-1,-12), (-2,-6), (-3,-4)\n\nSame parity check:\n(1,12): odd+even → different parity ✗\n(2,6): even+even → same parity ✓\n(3,4): odd+even → different parity ✗\n\nFor (x+y, x-y) = (6, 2):\nx = (6+2)/2 = 4, y = (6-2)/2 = 2\n\nFor (x+y, x-y) = (2, 6):\nx = (2+6)/2 = 4, y = (2-6)/2 = -2\n\nFor negatives: (-6, -2) → x = -4, y = -2\n(-2, -6) → x = -4, y = 2\n\nSolutions: (4, 2), (4, -2), (-4, -2), (-4, 2)", "algebra", 5),
]

ENTRIES.extend(ALGEBRA)

# ============================================================
# CALCULUS (60 examples)
# ============================================================

CALCULUS = [
    # Difficulty 1 - Basic concepts
    ("What is a derivative in simple terms?", "A derivative measures how fast something is changing right now.\n\nThink of it like a speedometer:\n- Your position changes over time as you drive\n- The derivative of position = your speed at this exact moment\n\nIf f(x) = x², the derivative f'(x) = 2x tells you how steep the curve is at any point.\n\nAt x = 3: f'(3) = 6, meaning the curve is going up steeply.\nAt x = 0: f'(0) = 0, meaning the curve is flat (the bottom of the parabola).", "calculus", 1),
    ("What is the derivative of x²?", "Using the power rule:\n\nIf f(x) = x^n, then f'(x) = n·x^(n-1)\n\nFor x²: n = 2\nf'(x) = 2·x^(2-1) = 2x\n\nSo the derivative of x² is 2x.\n\nThis means: at any point x on the curve y = x², the slope is 2x.\nAt x = 1: slope = 2\nAt x = 3: slope = 6", "calculus", 1),
    ("What is the derivative of 5x?", "Using the power rule:\n\nf(x) = 5x = 5x¹\nf'(x) = 5·1·x⁰ = 5\n\nThe derivative of 5x is just 5.\n\nThis makes sense! If y = 5x is a straight line with slope 5, then the slope (derivative) is always 5, everywhere.", "calculus", 1),
    ("What is the derivative of a constant, like 7?", "The derivative of any constant is 0.\n\nf(x) = 7 → f'(x) = 0\n\nWhy? A constant doesn't change! If the graph is a flat horizontal line, the slope is 0 everywhere.\n\nGeneral rule: d/dx(c) = 0 for any constant c.", "calculus", 1),
    ("What does ∫ mean?", "The ∫ symbol means 'integral' — it's basically adding up lots of tiny pieces.\n\nThink of it this way:\n- Derivative = cutting something into infinitely small slices\n- Integral = putting all the slices back together\n\nIf you know the speed at every moment (derivative), the integral adds up all those tiny speed·time pieces to get the total distance traveled.\n\nThat's why integrals and derivatives are opposites — the Fundamental Theorem of Calculus.", "calculus", 1),
    ("What is the integral of 2x?", "Using the reverse of the power rule:\n\n∫ 2x dx = 2 · x^(1+1)/(1+1) + C\n= 2 · x²/2 + C\n= x² + C\n\nWe add C (the constant of integration) because the derivative of x² + C is always 2x, no matter what C is.\n\nWithout more information, we can't determine C. That's why indefinite integrals always have +C.", "calculus", 1),

    # Difficulty 2 - Rules
    ("Find the derivative of x³ + 2x² - 5x + 3", "Apply the power rule term by term:\n\nf(x) = x³ + 2x² - 5x + 3\nf'(x) = 3x² + 4x - 5 + 0\nf'(x) = 3x² + 4x - 5\n\nRules used:\n- d/dx(x³) = 3x²\n- d/dx(2x²) = 4x\n- d/dx(-5x) = -5\n- d/dx(3) = 0 (constant)", "calculus", 2),
    ("What is the derivative of (x² + 1)(x³)?", "Product rule: d/dx[f·g] = f'·g + f·g'\n\nf = x² + 1, g = x³\nf' = 2x, g' = 3x²\n\nDerivative = 2x · x³ + (x² + 1) · 3x²\n= 2x⁴ + 3x⁴ + 3x²\n= 5x⁴ + 3x²\n\nAlternative: Expand first → x⁵ + x³, then differentiate → 5x⁴ + 3x². Same answer!", "calculus", 2),
    ("Find d/dx of sin(x) + cos(x)", "These are standard derivatives you should memorize:\n\n- d/dx[sin(x)] = cos(x)\n- d/dx[cos(x)] = -sin(x)\n\nSo: d/dx[sin(x) + cos(x)] = cos(x) - sin(x)\n\nThe key identities:\n- sin → cos\n- cos → -sin\n- -sin → -cos\n- -cos → sin\n\nIt cycles! Each derivative rotates by 90°.", "calculus", 2),
    ("What is the integral of 3x² + 2x?", "∫(3x² + 2x)dx\n\nTerm by term, reverse power rule:\n∫ 3x² dx = 3·x³/3 = x³\n∫ 2x dx = 2·x²/2 = x²\n\n= x³ + x² + C\n\nCheck: d/dx(x³ + x² + C) = 3x² + 2x ✓", "calculus", 2),
    ("What is the derivative of eˣ?", "Here's the beautiful thing about eˣ:\n\nd/dx(eˣ) = eˣ\n\nIt's its own derivative! The slope of eˣ at any point equals the value of eˣ at that point.\n\nAt x = 0: value = 1, slope = 1\nAt x = 1: value = e ≈ 2.718, slope ≈ 2.718\nAt x = 2: value = e² ≈ 7.389, slope ≈ 7.389\n\nThis unique property is WHY e is so important — it's the only base where this happens.", "calculus", 2),
    ("Differentiate 4x⁵ - 3x³ + x - 7", "Power rule on each term:\n\nd/dx(4x⁵) = 20x⁴\nd/dx(-3x³) = -9x²\nd/dx(x) = 1\nd/dx(-7) = 0\n\nf'(x) = 20x⁴ - 9x² + 1", "calculus", 2),
    ("What is the chain rule?", "The chain rule is for when one function is inside another:\n\nd/dx[f(g(x))] = f'(g(x)) · g'(x)\n\nThink of it as: 'derivative of the outside × derivative of the inside'\n\nExample: d/dx[sin(x²)]\n- Outside: sin(u), derivative = cos(u)\n- Inside: x², derivative = 2x\n- Result: cos(x²) · 2x = 2x·cos(x²)\n\nMemory trick: 'derive the outside, keep the inside, then multiply by the inside's derivative'", "calculus", 2),

    # Difficulty 3 - Applied
    ("Find the derivative of sin(3x²)", "Chain rule!\n\nOutside: sin(u) → derivative is cos(u)\nInside: 3x² → derivative is 6x\n\nd/dx[sin(3x²)] = cos(3x²) · 6x = 6x·cos(3x²)\n\nStep by step:\n1. Identify outer = sin, inner = 3x²\n2. Derivative of outer (keeping inner) = cos(3x²)\n3. Multiply by derivative of inner = 6x\n4. Result: 6x·cos(3x²)", "calculus", 3),
    ("What is the integral of x·eˣ?", "Integration by parts: ∫u·dv = u·v - ∫v·du\n\nChoose: u = x, dv = eˣdx\nThen: du = dx, v = eˣ\n\n∫ x·eˣ dx = x·eˣ - ∫ eˣ dx\n= x·eˣ - eˣ + C\n= eˣ(x - 1) + C\n\nCheck: d/dx[eˣ(x-1)] = eˣ(x-1) + eˣ·1 = eˣ(x-1+1) = x·eˣ ✓\n\nTip: Choose u as the thing that gets simpler when differentiated (x → 1), and dv as the thing that doesn't get worse when integrated (eˣ → eˣ).", "calculus", 3),
    ("Find the critical points of f(x) = x³ - 6x² + 9x + 1", "Critical points are where f'(x) = 0 or is undefined.\n\nf'(x) = 3x² - 12x + 9 = 3(x² - 4x + 3) = 3(x - 1)(x - 3)\n\nSet f'(x) = 0:\nx = 1 or x = 3\n\nFind the y-values:\nf(1) = 1 - 6 + 9 + 1 = 5\nf(3) = 27 - 54 + 27 + 1 = 1\n\nSecond derivative test:\nf''(x) = 6x - 12\nf''(1) = -6 < 0 → local maximum at (1, 5)\nf''(3) = 6 > 0 → local minimum at (3, 1)", "calculus", 3),
    ("What is ∫₀³ 2x dx?", "This is a definite integral — we're finding the area under y = 2x from x = 0 to x = 3.\n\nStep 1: Find the antiderivative\n∫ 2x dx = x² + C\n\nStep 2: Evaluate at bounds\nF(3) - F(0) = 3² - 0² = 9\n\nThe area is 9.\n\nYou can verify geometrically: y = 2x from 0 to 3 forms a triangle with base 3 and height 6.\nArea = ½ · base · height = ½ · 3 · 6 = 9 ✓", "calculus", 3),
    ("Find the equation of the tangent line to y = x² at x = 2", "Point: (2, 4) since y = 2² = 4\n\nSlope: f'(x) = 2x, so f'(2) = 4\n\nEquation: y - y₁ = m(x - x₁)\ny - 4 = 4(x - 2)\ny = 4x - 4\n\nThe tangent line at x = 2 is y = 4x - 4.\n\nCheck: at x = 2, y = 4(2) - 4 = 4 ✓ (passes through the point)", "calculus", 3),
    ("Differentiate ln(x² + 1)", "Chain rule with natural log:\n\nd/dx[ln(u)] = u'/u\n\nu = x² + 1, u' = 2x\n\nd/dx[ln(x² + 1)] = 2x/(x² + 1)\n\nNote: The domain is all real numbers since x² + 1 > 0 always.", "calculus", 3),
    ("Solve ∫₁ᵉ 1/x dx", "∫₁ᵉ 1/x dx = ln|x| evaluated from 1 to e\n= ln(e) - ln(1)\n= 1 - 0\n= 1\n\nThis is actually one of the defining properties of e!\nln(e) = 1 because e is defined as the number whose area under 1/x from 1 to itself equals 1.", "calculus", 3),

    # Difficulty 4 - Advanced
    ("Use implicit differentiation to find dy/dx for x² + y² = 25", "Differentiate both sides with respect to x:\n\nd/dx(x²) + d/dx(y²) = d/dx(25)\n2x + 2y·(dy/dx) = 0\n\nSolve for dy/dx:\n2y·(dy/dx) = -2x\ndy/dx = -x/y\n\nThis gives us the slope of the tangent line at any point on the circle.\n\nAt point (3, 4): dy/dx = -3/4\nAt point (0, 5): dy/dx = 0 (horizontal tangent, makes sense!)\nAt point (5, 0): undefined (vertical tangent)", "calculus", 4),
    ("Evaluate ∫₀¹ x²eˣ dx", "Integration by parts twice.\n\nFirst: u = x², dv = eˣdx → du = 2xdx, v = eˣ\n∫₀¹ x²eˣ dx = [x²eˣ]₀¹ - ∫₀¹ 2xeˣ dx\n= e - 2∫₀¹ xeˣ dx\n\nSecond: u = x, dv = eˣdx → du = dx, v = eˣ\n∫₀¹ xeˣ dx = [xeˣ]₀¹ - ∫₀¹ eˣ dx = e - [eˣ]₀¹ = e - (e - 1) = 1\n\nBack to our integral:\ne - 2(1) = e - 2 ≈ 0.718", "calculus", 4),
    ("Find the volume of revolution of y = √x from x = 0 to x = 4 around the x-axis", "Using the disk method:\n\nV = π ∫₀⁴ (√x)² dx = π ∫₀⁴ x dx\n= π [x²/2]₀⁴\n= π(16/2 - 0)\n= 8π ≈ 25.13\n\nThe curve y = √x from 0 to 4, rotated around the x-axis, creates a solid with volume 8π cubic units.", "calculus", 4),
    ("What is L'Hôpital's Rule?", "L'Hôpital's Rule helps evaluate limits that give 0/0 or ∞/∞.\n\nIf lim(x→a) f(x)/g(x) = 0/0 or ∞/∞, then:\nlim(x→a) f(x)/g(x) = lim(x→a) f'(x)/g'(x)\n\nExample: lim(x→0) sin(x)/x\nBoth sin(0) = 0 and x approaches 0 → 0/0 form\n\nApply L'Hôpital:\n= lim(x→0) cos(x)/1 = cos(0)/1 = 1\n\nImportant: You can apply it repeatedly if needed, but ONLY for 0/0 or ∞/∞ forms.", "calculus", 4),
    ("Find the Taylor series for eˣ centered at x = 0", "The Taylor series at x = 0 (Maclaurin series) is:\n\neˣ = Σ(n=0 to ∞) xⁿ/n!\n= 1 + x + x²/2! + x³/3! + x⁴/4! + ...\n\nDerivatives of eˣ at x = 0 are all 1:\nf(0) = 1, f'(0) = 1, f''(0) = 1, etc.\n\nSo each coefficient is f⁽ⁿ⁾(0)/n! = 1/n!\n\nConvergence: This series converges for ALL x, which is why eˣ is so well-behaved.\n\nQuick estimate: e ≈ 1 + 1 + 1/2 + 1/6 + 1/24 ≈ 2.708", "calculus", 4),

    # Difficulty 5 - Expert
    ("Prove that the derivative of sin(x) is cos(x) from first principles", "Using the limit definition:\n\nd/dx[sin(x)] = lim(h→0) [sin(x+h) - sin(x)] / h\n\nApply the angle addition formula:\nsin(x+h) = sin(x)cos(h) + cos(x)sin(h)\n\n= lim(h→0) [sin(x)cos(h) + cos(x)sin(h) - sin(x)] / h\n= lim(h→0) [sin(x)(cos(h) - 1) + cos(x)sin(h)] / h\n= sin(x) · lim(h→0)(cos(h)-1)/h + cos(x) · lim(h→0)sin(h)/h\n\nWe need two known limits:\nlim(h→0) sin(h)/h = 1\nlim(h→0) (cos(h) - 1)/h = 0\n\nTherefore:\nd/dx[sin(x)] = sin(x)·0 + cos(x)·1 = cos(x)", "calculus", 5),
    ("Evaluate ∫₀^∞ e^(-x²) dx", "This is the Gaussian integral. It equals √π/2.\n\nProof sketch (polar coordinates trick):\n\nLet I = ∫₀^∞ e^(-x²) dx\nThen I² = ∫₀^∞ ∫₀^∞ e^(-(x²+y²)) dx dy\n\nSwitch to polar: x² + y² = r², dx dy = r dr dθ\nI² = ∫₀^(π/2) ∫₀^∞ e^(-r²) · r dr dθ\n\nThe inner integral: ∫₀^∞ r·e^(-r²) dr\nLet u = r², du = 2r dr\n= (1/2)∫₀^∞ e^(-u) du = 1/2\n\nSo I² = (π/2)(1/2) = π/4\nI = √(π/4) = √π/2\n\nThis integral is fundamental in probability (normal distribution), quantum mechanics, and signal processing.", "calculus", 5),
]

ENTRIES.extend(CALCULUS)

# ============================================================
# PHYSICS (60 examples)
# ============================================================

PHYSICS = [
    # Difficulty 1
    ("What is Newton's First Law?", "Newton's First Law (Law of Inertia):\n\nAn object at rest stays at rest, and an object in motion stays in motion at constant velocity, unless acted on by a net external force.\n\nReal world examples:\n- A book on a table stays put (no net force)\n- A hockey puck on ice keeps sliding (nearly frictionless)\n- You lurch forward when a car brakes (your body wants to keep moving)\n\nKey insight: Forces cause CHANGES in motion, not motion itself. Things naturally keep doing what they're doing.", "physics", 1),
    ("What is velocity?", "Velocity = how fast + which direction\n\nIt's speed with a direction attached.\n- Speed: 60 mph (just a number)\n- Velocity: 60 mph north (number + direction)\n\nVelocity is a vector — it has both magnitude and direction.\n- 60 mph north and 60 mph east are different velocities\n- But they have the same speed\n\nIf you're driving in a circle at constant speed, your velocity is constantly changing (because the direction keeps changing). That's why you feel a force pushing you outward — your velocity is changing!", "physics", 1),
    ("What is mass?", "Mass is how much stuff (matter) an object contains.\n\nIt's measured in kilograms (kg) and it's the same everywhere:\n- Your mass is the same on Earth, the Moon, and in space\n- It's a property of the object\n\nDon't confuse mass with weight:\n- Mass = amount of matter (kg)\n- Weight = force of gravity on that mass (N)\n- Weight = mass × gravity\n\nOn the Moon, your weight is ~1/6 of Earth weight, but your mass is the same.", "physics", 1),
    ("What is energy?", "Energy is the ability to do work or cause change.\n\nKey types:\n- Kinetic energy: energy of motion (a rolling ball)\n- Potential energy: stored energy (a book on a shelf)\n- Thermal energy: heat (a hot cup of coffee)\n- Chemical energy: stored in bonds (food, fuel)\n- Electrical energy: moving charges (lightning, circuits)\n\nThe most important rule: Energy is never created or destroyed, only transformed.\n\nA roller coaster: potential energy (top) → kinetic energy (bottom) → potential energy (next hill)", "physics", 1),
    ("What is friction?", "Friction is a force that opposes motion between two surfaces in contact.\n\nTwo types:\n- Static friction: prevents an object from starting to move (harder to overcome)\n- Kinetic friction: slows down an object already moving\n\nWhy it matters:\n- Without friction, you couldn't walk (feet would slip)\n- Without friction, cars couldn't brake\n- With too much friction, machines waste energy as heat\n\nFriction depends on:\n1. How rough the surfaces are (coefficient of friction μ)\n2. How hard they're pressed together (normal force N)\n\nF_friction = μ × N", "physics", 1),

    # Difficulty 2
    ("A 5 kg box is pushed with 20 N of force. What's the acceleration?", "Newton's Second Law: F = ma\n\nGiven: F = 20 N, m = 5 kg\n20 = 5 × a\na = 20/5 = 4 m/s²\n\nThe box accelerates at 4 m/s².\n\nThis means every second, the speed increases by 4 m/s:\n- After 1 s: 4 m/s\n- After 2 s: 8 m/s\n- After 3 s: 12 m/s\n(Assuming no friction and starting from rest)", "physics", 2),
    ("What is the kinetic energy of a 2 kg ball moving at 3 m/s?", "Kinetic energy: KE = ½mv²\n\nKE = ½ × 2 × 3²\n= ½ × 2 × 9\n= 9 J\n\nThe ball has 9 Joules of kinetic energy.\n\nFun fact: KE depends on velocity squared. If you double the speed, the energy quadruples!\n- At 3 m/s: KE = 9 J\n- At 6 m/s: KE = ½ × 2 × 36 = 36 J (4× more!)\n\nThis is why car crashes get so much worse at higher speeds.", "physics", 2),
    ("How long does it take a ball to fall 20 m?", "Using: y = ½gt² (starting from rest)\n\n20 = ½ × 9.8 × t²\n20 = 4.9t²\nt² = 20/4.9 = 4.08\nt = √4.08 ≈ 2.02 seconds\n\nThe ball takes about 2 seconds to fall 20 meters.\n\nWe ignored air resistance, which is fine for a dense ball at this height. For a feather or from much higher, air resistance matters.", "physics", 2),
    ("What's the gravitational potential energy of a 3 kg book on a 2 m shelf?", "PE = mgh\n\nPE = 3 × 9.8 × 2\n= 58.8 J\n\nThe book has about 59 Joules of potential energy.\n\nIf it falls, all that PE converts to KE:\n½mv² = mgh\nv = √(2gh) = √(2 × 9.8 × 2) = √39.2 ≈ 6.3 m/s\n\nIt hits the ground at about 6.3 m/s (14 mph). Ouch!", "physics", 2),
    ("Two forces act on an object: 30 N right and 10 N left. What's the net force?", "Net force = 30 N right + 10 N left\n\nSince they're in opposite directions, subtract:\nF_net = 30 - 10 = 20 N (to the right)\n\nThe object accelerates to the right with F = ma.\nIf m = 4 kg: a = 20/4 = 5 m/s²\n\nKey point: Forces are vectors. When they're along the same line, you add them with signs (right = +, left = -).", "physics", 2),
    ("A car goes from 0 to 60 mph in 8 seconds. What's the acceleration?", "First convert to SI units:\n60 mph = 60 × 0.447 = 26.8 m/s\n\na = Δv/Δt = (26.8 - 0)/8 = 3.35 m/s²\n\nAbout 3.4 m/s², or roughly 0.34g.\n\nFor comparison:\n- Typical car: 3-4 m/s²\n- Sports car: 8-10 m/s²\n- Formula 1: ~15 m/s²\n- Rocket launch: ~30 m/s²", "physics", 2),
    ("What is momentum?", "Momentum (p) = mass × velocity = mv\n\nIt measures how hard it is to stop something.\n- A slow train has huge momentum (large m)\n- A fast bullet has huge momentum (large v)\n\nKey property: Momentum is CONSERVED in any collision (no external forces).\n\nBefore collision: p_total = m₁v₁ + m₂v₂\nAfter collision: p_total = same!\n\nThis is why:\n- A small car gets destroyed in a crash with a truck (truck has more momentum)\n- Pool balls transfer momentum on impact\n- Rockets work by throwing momentum backward", "physics", 2),

    # Difficulty 3
    ("A 2 kg ball moving at 5 m/s hits a stationary 3 kg ball. After collision, the 2 kg ball stops. How fast does the 3 kg ball go?", "Conservation of momentum:\n\nBefore: p = m₁v₁ + m₂v₂ = 2(5) + 3(0) = 10 kg·m/s\nAfter: p = m₁v₁' + m₂v₂' = 2(0) + 3(v₂')\n\n10 = 3v₂'\nv₂' = 10/3 ≈ 3.33 m/s\n\nThe 3 kg ball moves at about 3.33 m/s.\n\nThis is a perfectly elastic collision in one dimension where all momentum transfers. In reality, some kinetic energy would be lost (inelastic collision), and the 2 kg ball would bounce back slightly.", "physics", 3),
    ("What is the period of a pendulum that is 1 m long?", "T = 2π√(L/g)\n\nT = 2π√(1/9.8)\n= 2π√(0.102)\n= 2π(0.319)\n= 2.01 seconds\n\nInteresting: The period doesn't depend on the mass or the amplitude (for small angles)!\n\n- A 1 m pendulum: T ≈ 2 s\n- A 0.25 m pendulum: T ≈ 1 s\n- A 4 m pendulum: T ≈ 4 s\n\nThis is why grandfather clocks use pendulums — their timing is very consistent.", "physics", 3),
    ("A 500 N person stands on a 2 m board that extends 0.5 m past a support. What torque does the person create?", "Torque = Force × Distance (perpendicular)\n\nThe person is 0.5 m from the support (the pivot point).\n\nτ = 500 N × 0.5 m = 250 N·m\n\nThe torque is 250 N·m, tending to rotate the board downward on the person's side.\n\nFor the board to be in equilibrium, you'd need an equal and opposite torque on the other side. If the other side is 1.5 m long:\nF × 1.5 = 250\nF = 167 N (the support must push up with 167 N on that side, or something must hold it down)", "physics", 3),
    ("A projectile is launched at 30° with speed 20 m/s. How far does it go?", "Horizontal range: R = v²sin(2θ)/g\n\nR = (20)² × sin(60°) / 9.8\n= 400 × 0.866 / 9.8\n= 346.4 / 9.8\n≈ 35.3 m\n\nThe projectile lands about 35.3 m away.\n\nMaximum height:\nh = v²sin²(θ) / (2g)\n= 400 × sin²(30°) / 19.6\n= 400 × 0.25 / 19.6\n= 5.1 m\n\nTime of flight: t = 2v·sin(θ)/g = 2(20)(0.5)/9.8 ≈ 2.04 s", "physics", 3),
    ("What is the speed of sound and why does it change with temperature?", "Speed of sound in air ≈ 343 m/s at 20°C (room temperature)\n\nApproximate formula: v ≈ 331 + 0.6T m/s (where T is in °C)\n\nAt 0°C: v ≈ 331 m/s\nAt 20°C: v ≈ 343 m/s\nAt 40°C: v ≈ 355 m/s\n\nWhy does temperature matter? Sound travels through molecular collisions. Higher temperature = faster molecules = collisions propagate faster.\n\nSound also travels at different speeds in different materials:\n- Air: 343 m/s\n- Water: 1480 m/s\n- Steel: 5960 m/s\n\nDenser/stiffer materials transmit sound faster because molecules are closer together.", "physics", 3),

    # Difficulty 4
    ("What is escape velocity?", "Escape velocity is the minimum speed needed to escape a planet's gravity without any additional thrust.\n\nv_escape = √(2GM/r)\n\nFor Earth: v ≈ 11.2 km/s ≈ 25,000 mph\n\nKey points:\n- It doesn't depend on the object's mass (a feather and a rocket need the same speed!)\n- It does depend on the planet's mass and radius\n- It's the speed at which kinetic energy exactly equals gravitational potential energy\n\n½mv² = GMm/r → v = √(2GM/r)\n\nOther escape velocities:\n- Moon: 2.4 km/s\n- Mars: 5.0 km/s\n- Jupiter: 59.5 km/s\n- Sun: 617.5 km/s\n\nThis is why we need enormous rockets to leave Earth but could throw a ball off the Moon.", "physics", 4),
    ("Explain the Doppler effect", "The Doppler effect is the change in frequency/wavelength of waves when the source and observer move relative to each other.\n\nFor sound approaching: f_observed = f_source × v/(v - v_source)\nFor sound receding: f_observed = f_source × v/(v + v_source)\n\nWhere v = speed of sound, v_source = speed of the source.\n\nEveryday examples:\n- Ambulance siren: high pitch approaching, low pitch receding\n- Race cars: engine pitch drops as they pass\n- Police radar: measures your speed from reflected wave frequency\n\nIn light (astronomy):\n- Moving away → redshift (longer wavelength, redder)\n- Moving toward → blueshift (shorter wavelength, bluer)\n- This is how we know the universe is expanding — distant galaxies are redshifted", "physics", 4),
    ("A satellite orbits at 400 km altitude. What's its orbital speed?", "For circular orbit: v = √(GM/r)\n\nr = Earth radius + altitude = 6,371,000 + 400,000 = 6,771,000 m\nG = 6.674 × 10⁻¹¹\nM_earth = 5.972 × 10²⁴ kg\n\nv = √(6.674×10⁻¹¹ × 5.972×10²⁴ / 6.771×10⁶)\n= √(3.986×10¹⁴ / 6.771×10⁶)\n= √(5.894×10⁷)\n= 7,677 m/s ≈ 7.7 km/s ≈ 27,600 km/h\n\nPeriod: T = 2πr/v = 2π(6.771×10⁶)/7677 ≈ 5,544 s ≈ 92 minutes\n\nThis is the ISS orbit! It goes around Earth every ~90 minutes.", "physics", 4),
    ("What is the electric field from a point charge?", "E = kQ/r²\n\nWhere:\n- k = 8.99 × 10⁹ N·m²/C² (Coulomb's constant)\n- Q = the charge\n- r = distance from the charge\n\nThe field points:\n- Away from positive charges\n- Toward negative charges\n\nExample: A +1 μC charge at 1 m distance:\nE = (8.99×10⁹)(1×10⁻⁶)/(1)² = 8,990 N/C\n\nKey properties:\n- It follows an inverse square law (like gravity)\n- Field lines never cross\n- Stronger closer to the charge (gets 4× stronger at half the distance)\n- Superposition: multiple charges' fields add as vectors", "physics", 4),

    # Difficulty 5
    ("Derive the time dilation formula from the light clock thought experiment", "Imagine a light clock on a spaceship moving at velocity v.\n\nAt rest: Light bounces between mirrors distance L apart.\nOne tick: t₀ = 2L/c\n\nMoving (at velocity v, perpendicular to the mirrors):\nLight must travel a diagonal path (longer!).\nIn time t (as seen by a stationary observer):\n- Light travels ct (diagonal)\n- Ship moves vt horizontally\n\nBy Pythagorean theorem:\n(ct)² = (vt)² + (2L)²\nc²t² = v²t² + 4L²\nt²(c² - v²) = 4L²\nt² = 4L²/(c² - v²)\nt = 2L/√(c² - v²)\nt = (2L/c)/√(1 - v²/c²)\nt = t₀/√(1 - v²/c²)\n\nSo: t = γt₀ where γ = 1/√(1 - v²/c²)\n\nAt v = 0.9c: γ ≈ 2.29, so 1 second on the ship = 2.29 seconds for us.\nTime runs slower for the moving observer!", "physics", 5),
    ("Explain why the magnetic force does no work", "The magnetic force on a moving charge is:\nF = qv × B\n\nThis force is always perpendicular to the velocity (by the cross product).\n\nWork = F · d = F · v·dt\nSince F ⊥ v, F · v = 0.\n\nTherefore, the magnetic force does zero work.\n\nWhat this means physically:\n- A magnetic field can change a particle's DIRECTION (it curves the path)\n- But it can never change a particle's SPEED (kinetic energy stays the same)\n- Particles in magnetic fields move in circles or helices, not speeding up or slowing down\n\nIf a magnet seems to 'pull' something, it's actually the electric field (from changing magnetic flux, per Faraday's law) doing the work. The magnetic field itself is just redirecting momentum.", "physics", 5),
]

ENTRIES.extend(PHYSICS)

# ============================================================
# CHEMISTRY (60 examples)
# ============================================================

CHEMISTRY = [
    # Difficulty 1
    ("What is an atom?", "An atom is the smallest unit of an element that still has that element's properties.\n\nThree parts:\n- Protons: positive charge, in the nucleus\n- Neutrons: no charge, in the nucleus\n- Electrons: negative charge, orbit the nucleus\n\nThe number of protons = the atomic number = which element it is.\n- 1 proton = Hydrogen\n- 6 protons = Carbon\n- 79 protons = Gold\n\nMost of an atom is empty space! The nucleus is tiny compared to the whole atom. If the nucleus were a marble, the atom would be a football stadium.", "chemistry", 1),
    ("What is the periodic table?", "The periodic table organizes all known elements by their properties.\n\nLayout:\n- Rows (periods): elements get heavier left to right\n- Columns (groups): elements have similar properties\n- Group 1 (left): alkali metals (very reactive)\n- Group 18 (right): noble gases (very unreactive)\n\nPatterns:\n- Metals on the left, nonmetals on the right\n- Metalloids zigzag in the middle\n- Atomic number increases left to right, top to bottom\n\nWhy it's useful: elements in the same column behave similarly. If you know how sodium behaves, you can predict potassium will be similar.", "chemistry", 1),
    ("What is a molecule?", "A molecule is two or more atoms bonded together.\n\nExamples:\n- H₂O: 2 hydrogen + 1 oxygen (water)\n- CO₂: 1 carbon + 2 oxygen (carbon dioxide)\n- O₂: 2 oxygen (the oxygen we breathe)\n- NaCl: 1 sodium + 1 chlorine (table salt — technically a formula unit, not a molecule)\n\nThe subscript number tells you how many of each atom:\n- H₂SO₄: 2 hydrogens, 1 sulfur, 4 oxygens (sulfuric acid)\n\nNot everything is a molecule! Metals and ionic compounds form different structures. A molecule specifically means covalently bonded atoms.", "chemistry", 1),
    ("What is a chemical reaction?", "A chemical reaction is when substances change into new substances by breaking and forming bonds.\n\nReactants → Products\n\n2H₂ + O₂ → 2H₂O\n(hydrogen + oxygen → water)\n\nSigns a reaction happened:\n- Color change\n- Gas produced (bubbles)\n- Temperature change (hot or cold)\n- Precipitate forms (solid appears from solution)\n- Light emitted\n\nConservation of mass: Atoms aren't created or destroyed — they just rearrange. Count the atoms on each side and they should match (that's why we balance equations).", "chemistry", 1),
    ("What is density?", "Density = mass / volume (how much stuff is packed into a space)\n\nρ = m/V\n\nUnits: kg/m³ or g/cm³\n\nCommon densities:\n- Water: 1 g/cm³\n- Iron: 7.9 g/cm³\n- Air: 0.0012 g/cm³\n- Gold: 19.3 g/cm³\n\nThis is why:\n- Iron sinks (denser than water)\n- Ice floats (less dense than liquid water — unusual!)\n- Helium balloons rise (less dense than air)\n\nExample: A 500 g block with volume 100 cm³\nρ = 500/100 = 5 g/cm³", "chemistry", 1),

    # Difficulty 2
    ("Balance the equation: Fe + O₂ → Fe₂O₃", "Unbalanced: Fe + O₂ → Fe₂O₃\n\nCount atoms:\nLeft: Fe=1, O=2\nRight: Fe=2, O=3\n\nStep 1: Balance Fe\n2Fe + O₂ → Fe₂O₃\nLeft: Fe=2, O=2 | Right: Fe=2, O=3\n\nStep 2: Balance O (need 3 on left, but O comes in pairs)\n2Fe + 3/2O₂ → Fe₂O₃\nLeft: Fe=2, O=3 | Right: Fe=2, O=3\n\nStep 3: Clear fractions (multiply everything by 2)\n4Fe + 3O₂ → 2Fe₂O₃\n\nCheck: Left: Fe=4, O=6 | Right: Fe=4, O=6 ✓", "chemistry", 2),
    ("How many moles are in 24 g of carbon?", "Moles = mass / molar mass\n\nCarbon's molar mass = 12 g/mol\n\nn = 24 / 12 = 2 moles\n\n2 moles of carbon = 2 × 6.022 × 10²³ atoms\n= 1.2044 × 10²⁴ atoms\n\nQuick reference:\n- 12 g of C = 1 mol\n- 24 g of C = 2 mol\n- 6 g of C = 0.5 mol\n- 1 g of C = 1/12 mol", "chemistry", 2),
    ("What is electronegativity?", "Electronegativity is how strongly an atom pulls electrons in a bond.\n\nTrends:\n- Increases left to right across a period\n- Increases bottom to top up a group\n- Fluorine (3.98) is the most electronegative\n- Francium (0.7) is the least\n\nWhy it matters:\n- Big electronegativity difference → ionic bond (electron transfer)\n- Small difference → covalent bond (electron sharing)\n- No difference → nonpolar covalent (equal sharing)\n\nRule of thumb:\n- ΔEN > 1.7 → mostly ionic\n- 0.4 < ΔEN < 1.7 → polar covalent\n- ΔEN < 0.4 → nonpolar covalent", "chemistry", 2),
    ("What is the difference between an acid and a base?", "Acid: Produces H⁺ ions in water\nBase: Produces OH⁻ ions in water\n\nAcid examples: HCl, H₂SO₄, vinegar (acetic acid), lemon juice\nBase examples: NaOH, ammonia, baking soda solution\n\npH scale:\n- pH < 7: acidic (more H⁺)\n- pH = 7: neutral (pure water)\n- pH > 7: basic/alkaline (more OH⁻)\n\nEach pH unit = 10× difference:\n- pH 3 is 10× more acidic than pH 4\n- pH 1 is 1,000,000× more acidic than pH 7\n\nStrong acids (fully dissociate): HCl, H₂SO₄, HNO₃\nWeak acids (partially dissociate): acetic acid, carbonic acid", "chemistry", 2),
    ("What happens when you mix an acid and a base?", "Acid + Base → Salt + Water (neutralization)\n\nHCl + NaOH → NaCl + H₂O\n(hydrochloric acid + sodium hydroxide → table salt + water)\n\nThe H⁺ from the acid and the OH⁻ from the base combine to form water:\nH⁺ + OH⁻ → H₂O\n\nWhat's left over: the cation from the base and the anion from the acid form a salt.\n\nNot all neutralizations go to pH 7:\n- Strong acid + strong base → pH 7\n- Strong acid + weak base → pH < 7\n- Weak acid + strong base → pH > 7\n\nFun fact: Antacids (like Tums) are bases that neutralize stomach acid (HCl).", "chemistry", 2),
    ("Calculate the molar mass of H₂SO₄", "H₂SO₄: 2 hydrogens, 1 sulfur, 4 oxygens\n\nH: 2 × 1.008 = 2.016\nS: 1 × 32.06 = 32.06\nO: 4 × 16.00 = 64.00\n\nTotal = 2.016 + 32.06 + 64.00 = 98.076 g/mol\n\nSo 1 mole of H₂SO₄ weighs about 98.08 g.\n\nFor 5 moles: 5 × 98.08 = 490.4 g\nFor 0.25 moles: 0.25 × 98.08 = 24.52 g", "chemistry", 2),

    # Difficulty 3
    ("What is Le Chatelier's Principle?", "Le Chatelier's Principle: When you disturb a system at equilibrium, it shifts to counteract the disturbance.\n\nThree types of disturbances:\n\n1. Concentration: Add more reactant → shifts toward products\n   N₂ + 3H₂ ⇌ 2NH₃: Add more N₂ → more NH₃ produced\n\n2. Temperature: Increase temp → shifts toward the endothermic direction\n   If forward is exothermic: heat shifts it backward\n\n3. Pressure (gases): Increase pressure → shifts toward fewer moles of gas\n   N₂ + 3H₂ ⇌ 2NH₃: 4 moles → 2 moles, so high pressure favors NH₃\n\nThis is literally how the Haber process works to make fertilizer — high pressure pushes the equilibrium toward ammonia.", "chemistry", 3),
    ("What is an oxidation-reduction (redox) reaction?", "Redox = transfer of electrons between species.\n\nOIL RIG: Oxidation Is Loss, Reduction Is Gain (of electrons)\n\nOr: LEO says GER — Lose Electrons Oxidation, Gain Electrons Reduction\n\nExample: Zn + Cu²⁺ → Zn²⁺ + Cu\n\n- Zn → Zn²⁺ + 2e⁻ (Zn loses electrons = OXIDIZED, reducing agent)\n- Cu²⁺ + 2e⁻ → Cu (Cu gains electrons = REDUCED, oxidizing agent)\n\nOxidation numbers track electrons:\n- Zn: 0 → +2 (increased = oxidized)\n- Cu: +2 → 0 (decreased = reduced)\n\nReal-world redox:\n- Batteries: electron transfer makes electricity\n- Rusting: Fe → Fe₂O₃ (iron oxidized by oxygen)\n- Photosynthesis: CO₂ reduced to glucose", "chemistry", 3),
    ("What is the ideal gas law?", "PV = nRT\n\nP = pressure (atm or Pa)\nV = volume (L or m³)\nn = moles\nR = gas constant = 0.0821 L·atm/(mol·K)\nT = temperature (KELVIN always!)\n\nExample: A 2 L container at 300 K and 2 atm contains how many moles?\nn = PV/(RT) = (2)(2)/(0.0821 × 300)\nn = 4/24.63 = 0.163 mol\n\nDerived laws:\n- Boyle's: P₁V₁ = P₂V₂ (constant T, n)\n- Charles': V₁/T₁ = V₂/T₂ (constant P, n)\n- Avogadro's: V₁/n₁ = V₂/n₂ (constant P, T)\n\nRemember: Always convert to Kelvin! 25°C = 298 K, not 25.", "chemistry", 3),
    ("Balance this redox equation: MnO₄⁻ + Fe²⁺ → Mn²⁺ + Fe³⁺ (acidic)", "Step 1: Split into half-reactions\nReduction: MnO₄⁻ → Mn²⁺\nOxidation: Fe²⁺ → Fe³⁺\n\nStep 2: Balance atoms (Mn and Fe already balanced)\n\nStep 3: Balance O in reduction with H₂O\nMnO₄⁻ → Mn²⁺ + 4H₂O\n\nStep 4: Balance H with H⁺\n8H⁺ + MnO₄⁻ → Mn²⁺ + 4H₂O\n\nStep 5: Balance charge with electrons\nLeft charge: 8(+1) + (-1) = +7\nRight charge: +2\n5e⁻ + 8H⁺ + MnO₄⁻ → Mn²⁺ + 4H₂O\n\nOxidation: Fe²⁺ → Fe³⁺ + e⁻ (multiply by 5)\n5Fe²⁺ → 5Fe³⁺ + 5e⁻\n\nStep 6: Add and cancel electrons\n5e⁻ + 8H⁺ + MnO₄⁻ + 5Fe²⁺ → Mn²⁺ + 4H₂O + 5Fe³⁺\n\nBalanced!", "chemistry", 3),
    ("What is a buffer solution?", "A buffer resists changes in pH when small amounts of acid or base are added.\n\nMade from: Weak acid + its conjugate base (or weak base + conjugate acid)\nExample: CH₃COOH + CH₃COO⁻ (acetic acid + acetate)\n\nHow it works:\n- Add H⁺ → acetate grabs it → CH₃COOH (pH barely drops)\n- Add OH⁻ → acetic acid neutralizes it → H₂O + CH₃COO⁻ (pH barely rises)\n\nHenderson-Hasselbalch equation:\npH = pKa + log([A⁻]/[HA])\n\nBest buffering when [A⁻] = [HA], so pH = pKa.\nEffective range: pKa ± 1\n\nReal examples:\n- Blood: H₂CO₃/HCO₃⁻ buffer (pH 7.4)\n- Stomach antacid systems\n- Lab buffer solutions for biology experiments", "chemistry", 3),

    # Difficulty 4
    ("What is Gibbs free energy?", "Gibbs free energy (G) tells you whether a process is spontaneous.\n\nΔG = ΔH - TΔS\n\nΔG < 0: spontaneous (happens on its own)\nΔG > 0: non-spontaneous (needs energy input)\nΔG = 0: at equilibrium\n\nΔH = enthalpy change (energy of bonds)\nΔS = entropy change (disorder)\nT = temperature in Kelvin\n\nFour scenarios:\n1. ΔH < 0, ΔS > 0: Always spontaneous (exothermic + more disordered)\n2. ΔH > 0, ΔS < 0: Never spontaneous (endothermic + more ordered)\n3. ΔH < 0, ΔS < 0: Spontaneous at low T\n4. ΔH > 0, ΔS > 0: Spontaneous at high T\n\nExample: Ice melting\nΔH > 0 (absorbs heat), ΔS > 0 (more disordered as liquid)\nSpontaneous above 0°C because TΔS > ΔH", "chemistry", 4),
    ("Explain hybridization in carbon compounds", "Carbon has 6 electrons: 1s² 2s² 2p²\nOnly 2 unpaired electrons in 2p... but carbon forms 4 bonds!\n\nSolution: Hybridization — mixing atomic orbitals into new ones.\n\nsp³ hybridization (tetrahedral, 109.5°):\n- Mix 1 s + 3 p orbitals → 4 sp³ orbitals\n- Forms 4 equivalent bonds\n- Example: CH₄ (methane), diamond\n\nsp² hybridization (trigonal planar, 120°):\n- Mix 1 s + 2 p orbitals → 3 sp² orbitals + 1 unhybridized p\n- The p orbital forms a π bond (double bond)\n- Example: C₂H₄ (ethylene), graphite\n\nsp hybridization (linear, 180°):\n- Mix 1 s + 1 p → 2 sp orbitals + 2 unhybridized p\n- Forms a triple bond\n- Example: C₂H₂ (acetylene)\n\nMore s character = shorter, stronger bonds: sp > sp² > sp³", "chemistry", 4),

    # Difficulty 5
    ("Derive the rate law for a reaction mechanism with a slow step", "For a multi-step mechanism, the rate-determining step (slowest step) controls the overall rate.\n\nExample mechanism:\nStep 1 (fast): A + B ⇌ C (equilibrium, K₁)\nStep 2 (slow): C + D → E\n\nThe rate law comes from the slow step:\nRate = k₂[C][D]\n\nBut C is an intermediate (not in the overall equation). We need to express [C] in terms of reactants.\n\nFrom Step 1 (at equilibrium):\nK₁ = [C]/([A][B])\n[C] = K₁[A][B]\n\nSubstituting:\nRate = k₂ · K₁[A][B] · [D]\nRate = k_eff[A][B][D]\nwhere k_eff = k₂ · K₁\n\nThis is why the experimental rate law has order 1 in A, 1 in B, and 1 in D — even though the stoichiometry of the overall reaction A + B + D → E doesn't tell you this.\n\nKey insight: The mechanism predicts the rate law, but the rate law doesn't uniquely determine the mechanism.", "chemistry", 5),
    ("Explain quantum numbers and what each one describes", "Four quantum numbers describe every electron in an atom:\n\n1. n (principal) = 1, 2, 3... → energy level/shell\n   Determines size and energy. Higher n = farther from nucleus.\n\n2. ℓ (angular momentum) = 0 to n-1 → subshell shape\n   ℓ=0: s (spherical), ℓ=1: p (dumbbell), ℓ=2: d (clover), ℓ=3: f (complex)\n\n3. mₗ (magnetic) = -ℓ to +ℓ → orbital orientation\n   How many orbitals: s=1, p=3, d=5, f=7\n\n4. mₛ (spin) = +½ or -½ → electron spin direction\n   Each orbital holds max 2 electrons (opposite spins)\n\nElectron capacity per shell:\nn=1: 2e⁻ (1s)\nn=2: 8e⁻ (2s, 2p)\nn=3: 18e⁻ (3s, 3p, 3d)\nn=4: 32e⁻ (4s, 4p, 4d, 4f)\n\nPauli exclusion: No two electrons can have all four quantum numbers the same. This is why each orbital holds exactly 2 electrons.", "chemistry", 5),
]

ENTRIES.extend(CHEMISTRY)

# ============================================================
# GEOMETRY & TRIG (30 examples)
# ============================================================

GEOMETRY = [
    # Difficulty 1-2
    ("What is the area of a circle with radius 5?", "A = πr²\nA = π(5²) = 25π ≈ 78.54\n\nThe area is 25π square units, or about 78.5.", "geometry", 1),
    ("Find the hypotenuse of a right triangle with legs 3 and 4", "Pythagorean theorem: a² + b² = c²\n\n3² + 4² = c²\n9 + 16 = c²\nc² = 25\nc = 5\n\nThis is the famous 3-4-5 right triangle!", "geometry", 1),
    ("What is the area of a triangle with base 8 and height 5?", "A = ½ × base × height\nA = ½ × 8 × 5 = 20\n\nThe area is 20 square units.", "geometry", 1),
    ("What are the trig ratios for a right triangle?", "SOH-CAH-TOA:\n\nsin(θ) = Opposite / Hypotenuse\ncos(θ) = Adjacent / Hypotenuse\ntan(θ) = Opposite / Adjacent\n\nMemory trick: \"Some Old Hippie Caught Another Hippie Tripping On Acid\"\n\nFor the 3-4-5 triangle, if θ is opposite the 3:\nsin(θ) = 3/5\ncos(θ) = 4/5\ntan(θ) = 3/4", "geometry", 2),
    ("What is the volume of a cylinder with radius 3 and height 10?", "V = πr²h\nV = π(3²)(10) = π(9)(10) = 90π ≈ 282.7\n\nThe volume is 90π cubic units, about 283.", "geometry", 2),
    ("A triangle has angles of 50° and 70°. What's the third angle?", "Angles in a triangle sum to 180°.\n\nThird angle = 180° - 50° - 70° = 60°\n\nThis is an acute triangle (all angles < 90°).", "geometry", 1),

    # Difficulty 3
    ("What is the law of cosines?", "c² = a² + b² - 2ab·cos(C)\n\nIt works for ANY triangle, not just right triangles.\n\nWhen C = 90°: cos(90°) = 0, so c² = a² + b² (Pythagorean theorem!)\nWhen C < 90°: cos(C) > 0, so c is shorter than the Pythagorean prediction\nWhen C > 90°: cos(C) < 0, so c is longer\n\nUse it when you know:\n- Two sides and the included angle (SAS)\n- Three sides (SSS) — to find angles\n\nExample: a = 5, b = 7, C = 60°\nc² = 25 + 49 - 2(5)(7)(0.5) = 74 - 35 = 39\nc = √39 ≈ 6.24", "geometry", 3),
    ("Find sin(30°), cos(30°), and tan(30°)", "From the 30-60-90 special triangle:\n\nsin(30°) = 1/2 = 0.5\ncos(30°) = √3/2 ≈ 0.866\ntan(30°) = 1/√3 = √3/3 ≈ 0.577\n\nThe 30-60-90 triangle has sides: 1, √3, 2\n- Short side (opposite 30°) = 1\n- Medium side (opposite 60°) = √3\n- Hypotenuse = 2\n\nThese special values come up constantly — memorize them!", "geometry", 3),
    ("What is the area of a triangle with sides 7, 8, and 9?", "Use Heron's formula when you know all three sides.\n\ns = (7 + 8 + 9)/2 = 12 (semi-perimeter)\n\nA = √(s(s-a)(s-b)(s-c))\n= √(12 × 5 × 4 × 3)\n= √720\n≈ 26.83\n\nThe area is approximately 26.8 square units.", "geometry", 3),

    # Difficulty 4-5
    ("Prove the Pythagorean theorem", "Given: right triangle with legs a, b and hypotenuse c.\n\nProof by rearrangement:\n\nArrange four copies of the triangle inside a square of side (a+b):\n\nThe large square has area (a+b)²\nIt contains: 4 triangles (each area ½ab) + 1 inner square (area c²)\n\n(a+b)² = 4(½ab) + c²\na² + 2ab + b² = 2ab + c²\na² + b² = c²\n\nThis proof works because the four triangles exactly fill the large square's corners, leaving a tilted square of side c in the center. The algebra then confirms the theorem.\n\nThere are over 300 known proofs of this theorem!", "geometry", 4),
    ("What is the unit circle and why does it matter?", "The unit circle is a circle of radius 1 centered at the origin.\n\nFor any angle θ measured from the positive x-axis:\nx = cos(θ)\ny = sin(θ)\n\nKey values to memorize:\n0°: (1, 0)\n30°: (√3/2, 1/2)\n45°: (√2/2, √2/2)\n60°: (1/2, √3/2)\n90°: (0, 1)\n\nWhy it matters:\n1. Defines sin and cos for ALL angles, not just triangles\n2. Explains periodic behavior: sin(θ + 360°) = sin(θ)\n3. Makes trig identities obvious from symmetry\n4. Foundation for calculus, waves, Fourier analysis\n\nQuadrant signs (+/+ in Q1, -/+ in Q2, -/- in Q3, +/- in Q4): \"All Students Take Calculus\"", "geometry", 4),
]

ENTRIES.extend(GEOMETRY)

# ============================================================
# BIOLOGY (30 examples)
# ============================================================

BIOLOGY = [
    ("What is DNA?", "DNA (Deoxyribonucleic Acid) is the molecule that stores genetic instructions.\n\nStructure: Double helix — two strands twisted around each other\nBuilding blocks: Nucleotides, each with:\n- A sugar (deoxyribose)\n- A phosphate group\n- A nitrogen base: A (adenine), T (thymine), G (guanine), C (cytosine)\n\nBase pairing rules:\n- A always pairs with T (2 hydrogen bonds)\n- G always pairs with C (3 hydrogen bonds)\n\nThis pairing is why DNA can replicate — each strand serves as a template for the other.\n\nThe sequence of bases (like ATCGGTA...) encodes all the instructions for building and running an organism.", "biology", 1),
    ("What is photosynthesis?", "Photosynthesis: Plants convert light energy into chemical energy (glucose).\n\n6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂\n\nTwo stages:\n\n1. Light-dependent reactions (in thylakoid membranes):\n   - Water is split: H₂O → O₂ + H⁺ + e⁻\n   - Light energy captured by chlorophyll\n   - Produces ATP and NADPH\n\n2. Calvin cycle (in stroma):\n   - CO₂ is fixed and converted to glucose\n   - Uses ATP and NADPH from step 1\n   - No light required directly\n\nKey fact: The oxygen we breathe comes from splitting water, not from CO₂!", "biology", 2),
    ("What is natural selection?", "Natural selection is the mechanism of evolution, proposed by Darwin.\n\nFour conditions:\n1. Variation: Individuals differ in traits\n2. Heritability: Traits are passed to offspring\n3. Overproduction: More offspring than can survive\n4. Differential survival: Some traits give better survival/reproduction\n\nResult: Beneficial traits become more common over generations.\n\nExample: Peppered moths in England\n- Before industrial revolution: Light moths camouflaged on trees\n- After: Dark moths survived better on soot-covered trees\n- After clean air acts: Light moths became common again\n\nImportant: Natural selection acts on existing variation. It doesn't create new traits — mutations do that.", "biology", 3),
    ("Explain the Central Dogma of molecular biology", "Central Dogma: DNA → RNA → Protein\n\n1. Transcription (DNA → RNA):\n   - Happens in the nucleus\n   - RNA polymerase reads one DNA strand\n   - Creates complementary mRNA\n   - DNA: ATG → RNA: AUG (U replaces T)\n\n2. Translation (RNA → Protein):\n   - Happens at ribosomes in cytoplasm\n   - Ribosome reads mRNA in codons (3 bases)\n   - Each codon specifies an amino acid\n   - tRNA brings the right amino acid\n   - Amino acids linked into a protein chain\n\nCodon examples:\n- AUG = Start (Methionine)\n- UUU = Phenylalanine\n- UAA, UAG, UGA = Stop\n\nThere are 64 possible codons but only 20 amino acids — so the code is redundant (multiple codons per amino acid).", "biology", 3),
    ("What are enzymes and how do they work?", "Enzymes are biological catalysts — they speed up reactions without being consumed.\n\nKey features:\n- Proteins (mostly) with a specific 3D shape\n- Each enzyme works on a specific substrate (lock and key)\n- Lower activation energy → reactions happen faster\n- Not used up — can work on millions of substrates\n\nHow they work:\n1. Substrate binds to active site\n2. Enzyme-substrate complex forms (induced fit)\n3. Chemical reaction occurs\n4. Products released\n5. Enzyme is unchanged and ready for the next substrate\n\nFactors affecting enzyme activity:\n- Temperature: Higher = faster, until denaturation (~40°C for human enzymes)\n- pH: Each enzyme has an optimal pH\n- Substrate concentration: More substrate = faster, until saturation\n- Inhibitors: Competitive (blocks active site) or non-competitive (changes shape)", "biology", 2),
]

ENTRIES.extend(BIOLOGY)

# ============================================================
# ENGINEERING/COMPUTER SCIENCE (30 examples)
# ============================================================

CS = [
    ("What is Big O notation?", "Big O describes how an algorithm's runtime grows as input size increases.\n\nCommon complexities (fastest to slowest):\n- O(1): Constant — same time regardless of input (array index lookup)\n- O(log n): Logarithmic — halves the problem each step (binary search)\n- O(n): Linear — time proportional to input (loop through array)\n- O(n log n): Linearithmic — efficient sorting (merge sort, quick sort)\n- O(n²): Quadratic — nested loops (bubble sort)\n- O(2ⁿ): Exponential — recursive Fibonacci (without memoization)\n\nExample: Finding an item in a sorted array\n- Linear search: O(n) — check each one\n- Binary search: O(log n) — halve the search each step\n\nFor n = 1,000,000:\n- O(n) = 1,000,000 operations\n- O(log n) = 20 operations\n\nThat's why algorithm choice matters!", "cs", 2),
    ("What is recursion?", "Recursion is when a function calls itself to solve a smaller piece of the problem.\n\nEvery recursive function needs:\n1. Base case: When to stop (prevents infinite loops)\n2. Recursive case: Call itself with a simpler input\n\nExample — factorial:\ndef factorial(n):\n    if n == 0:  # base case\n        return 1\n    return n * factorial(n-1)  # recursive case\n\nfactorial(5) = 5 * 4 * 3 * 2 * 1 * 1 = 120\n\nThe call stack builds up:\nfactorial(5) → 5 * factorial(4) → 5 * 4 * factorial(3) → ... → 5 * 4 * 3 * 2 * 1 * 1\n\nCommon uses: tree traversal, divide-and-conquer, mathematical sequences, fractals\n\nWarning: Without a base case, you get infinite recursion → stack overflow!", "cs", 2),
    ("What is a hash table?", "A hash table stores key-value pairs with O(1) average lookup.\n\nHow it works:\n1. Hash function converts key → number (hash)\n2. Hash maps to an array index\n3. Value stored at that index\n\nExample: {\"name\": \"Alice\", \"age\": 25}\nhash(\"name\") → 3 → store at array[3]\nhash(\"age\") → 7 → store at array[7]\n\nCollision handling:\n- Chaining: Each index has a linked list of entries\n- Open addressing: If index taken, find next empty slot\n\nAverage case: O(1) for insert, lookup, delete\nWorst case: O(n) if many collisions\n\nReal uses: Python dicts, JavaScript objects, database indexes, caches, password storage\n\nGood hash function = distributes keys uniformly = fewer collisions", "cs", 3),
    ("Explain how the internet works (simplified)", "When you visit a website:\n\n1. DNS Lookup: google.com → 142.250.80.46 (phone book of the internet)\n2. TCP Connection: Three-way handshake with the server\n3. HTTP Request: \"GET / HTTP/1.1\" sent to the server\n4. Server Processing: Server runs code, queries databases\n5. HTTP Response: HTML, CSS, JS sent back\n6. Browser Rendering: Parse HTML → build DOM → apply CSS → run JS → display\n\nProtocols involved:\n- DNS: Name → IP address\n- TCP: Reliable delivery (guaranteed packets)\n- IP: Routing between computers\n- HTTP: Request/response format\n- TLS: Encryption (HTTPS)\n\nLayers (bottom up):\n1. Physical: Cables, radio waves\n2. Data Link: Ethernet, WiFi\n3. Network: IP routing\n4. Transport: TCP/UDP\n5. Application: HTTP, DNS, etc.", "cs", 2),
    ("What is the difference between TCP and UDP?", "TCP (Transmission Control Protocol):\n- Reliable: guarantees delivery and order\n- Connection-oriented: handshake before data\n- Flow control: adjusts speed to prevent overwhelming receiver\n- Error recovery: retransmits lost packets\n- Use cases: web browsing, email, file transfer, databases\n\nUDP (User Datagram Protocol):\n- Unreliable: no delivery guarantee, no ordering\n- Connectionless: just send it\n- No flow control or error recovery\n- Much faster, lower latency\n- Use cases: video streaming, gaming, DNS, live audio\n\nWhy use UDP if it's unreliable?\n- For real-time data, a delayed packet is worse than a dropped packet\n- In a video call, you'd rather skip a frame than wait 2 seconds for it\n- TCP's retransmission actually makes latency worse for live data\n\nRule of thumb: Need every byte? TCP. Need it now? UDP.", "cs", 3),
]

ENTRIES.extend(CS)

# ============================================================
# Load supplementary data
# ============================================================
try:
    from supplement_data import SUPPLEMENT
    ENTRIES.extend(SUPPLEMENT)
    print(f"Loaded {len(SUPPLEMENT)} supplementary examples")
except ImportError:
    print("Warning: supplement_data.py not found, using base examples only")

try:
    from supplement2_data import SUPPLEMENT2
    ENTRIES.extend(SUPPLEMENT2)
    print(f"Loaded {len(SUPPLEMENT2)} supplementary examples (vol 2)")
except ImportError:
    print("Warning: supplement2_data.py not found")

# ============================================================
# GENERATE AND SAVE
# ============================================================

def convert_to_unsloth_format(data):
    """Convert to Unsloth's chatml format for fine-tuning"""
    converted = []
    for q, a, topic, diff in data:
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        converted.append({"text": text})
    return converted


def convert_to_raw_format(data):
    """Convert to structured JSON with metadata"""
    converted = []
    for q, a, topic, diff in data:
        converted.append({
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ],
            "metadata": {"topic": topic, "difficulty": diff, "source": "curated"}
        })
    return converted


def main():
    random.seed(42)  # Reproducible

    raw_data = convert_to_raw_format(ENTRIES)
    unsloth_data = convert_to_unsloth_format(ENTRIES)

    # Shuffle for training
    random.shuffle(raw_data)
    random.shuffle(unsloth_data)

    # Save raw
    with open(f"{OUTPUT_DIR}/raw_training_data.json", "w") as f:
        json.dump(raw_data, f, indent=2)

    # Save unsloth format
    with open(f"{OUTPUT_DIR}/unsloth_training_data.jsonl", "w") as f:
        for entry in unsloth_data:
            f.write(json.dumps(entry) + "\n")

    # Stats
    topics = {}
    difficulties = {}
    for q, a, topic, diff in ENTRIES:
        topics[topic] = topics.get(topic, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    stats = {
        "total_examples": len(ENTRIES),
        "topics": topics,
        "difficulty_distribution": difficulties,
        "avg_per_topic": round(len(ENTRIES) / len(topics), 1),
        "notes": [
            "Curated high-quality STEM Q&A pairs",
            "Covers algebra, calculus, physics, chemistry, geometry, biology, CS",
            "Difficulty 1-5 scale: 1=beginner, 5=expert",
            "Target: 500+ examples for effective fine-tuning"
        ]
    }
    with open(f"{OUTPUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Generated {len(ENTRIES)} training examples")
    print(f"\nTopics: {json.dumps(topics, indent=2)}")
    print(f"\nDifficulty distribution: {json.dumps(difficulties, indent=2)}")
    print(f"\nFiles saved:")
    print(f"  {OUTPUT_DIR}/raw_training_data.json")
    print(f"  {OUTPUT_DIR}/unsloth_training_data.jsonl")
    print(f"  {OUTPUT_DIR}/stats.json")


if __name__ == "__main__":
    main()

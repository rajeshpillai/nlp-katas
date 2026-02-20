# Demonstrate Sequence Modeling Challenges

> Phase 8 — Context & Sequence Modeling | Kata 8.3

---

## Concept & Intuition

### What problem are we solving?

In Kata 8.1, we showed that word order carries meaning and BoW throws it away. In Kata 8.2, we showed that context resolves ambiguity and fixed representations miss it. The natural next step is: **build a model that processes text as a sequence**, updating its internal state word by word, so that each position carries information about everything that came before.

This is exactly what Recurrent Neural Networks (RNNs) do. An RNN reads one token at a time, maintains a hidden state vector, and updates that state at each step. The state at position t is a function of the current word and the state at position t-1. In theory, the state at the final position encodes information about the entire sequence.

But RNNs have a critical flaw: **information from early tokens fades as the sequence grows**. Each state update involves multiplying by a weight matrix, and when these multiplications are repeated many times, the contribution of early tokens either vanishes (shrinks to zero) or explodes (grows uncontrollably). This is called the **vanishing/exploding gradient problem**, and it means RNNs struggle with **long-range dependencies** -- relating words that are far apart in a sequence.

This kata demonstrates these challenges with concrete numerical examples, using plain Python arithmetic to show how repeated multiplication causes signals to decay, and why this motivated the invention of attention mechanisms.

### Why earlier methods fail

**BoW and TF-IDF** do not process sequences at all. They produce a single unordered vector for the entire document. No temporal processing means no ability to track how meaning builds through a sentence.

**N-grams** capture local order in fixed windows, but as shown in Kata 8.1, they cannot capture long-range dependencies and suffer from exponential vocabulary growth.

**Simple windowed averaging** (Kata 8.2) is not truly sequential. It looks at a fixed window around each token but does not accumulate information across the entire sequence. Position 50 in a document cannot "see" position 1 unless the window is 50 words wide.

**RNNs** were designed to solve this: process text sequentially, carry information forward through a hidden state. But the vanishing gradient problem means they fail at the very thing they were designed to do -- maintaining long-range memory. This is not a minor technical issue; it is a fundamental mathematical limitation of sequential state updates through multiplication.

### Mental models

- **An RNN is like a game of telephone.** Each person (token position) receives a message (hidden state), modifies it slightly, and passes it on. After enough people, the original message is lost. Early information fades because it passes through too many transformations.
- **Vanishing gradients = forgetting.** When you multiply a number less than 1 by itself many times, it shrinks toward zero. If the hidden state is scaled down at each step, contributions from early tokens become negligible by the end of a long sequence.
- **Exploding gradients = instability.** When you multiply a number greater than 1 by itself many times, it grows without bound. The model's hidden state becomes numerically unstable, and outputs become meaningless.
- **Attention is direct access.** Instead of passing information through a chain of intermediate states, attention lets any position directly look at any other position. Position 50 can directly attend to position 1 without the signal passing through 49 intermediate transformations.

### Visual explanations

```
HOW AN RNN PROCESSES A SEQUENCE
================================

  Input: "The cat sat on the mat"
  Tokens: [the, cat, sat, on, the, mat]

  h_0 = [0, 0, 0]                (initial hidden state)

  Step 1: h_1 = f(h_0, "the")    (update state with first word)
  Step 2: h_2 = f(h_1, "cat")    (update state with second word)
  Step 3: h_3 = f(h_2, "sat")    (update state with third word)
  Step 4: h_4 = f(h_3, "on")     (update state with fourth word)
  Step 5: h_5 = f(h_4, "the")    (update state with fifth word)
  Step 6: h_6 = f(h_5, "mat")    (final state -- should encode whole sentence)

  Each h_t depends on ALL previous tokens... in theory.
  In practice, h_6 is dominated by recent tokens ("the", "mat")
  and has nearly forgotten early tokens ("the", "cat").


THE VANISHING SIGNAL PROBLEM
==============================

  Simplified: h_t = w * h_{t-1}   (multiply by weight w at each step)

  If w = 0.9 (slightly less than 1):

    Step  1: 0.9^1  = 0.9000
    Step  5: 0.9^5  = 0.5905
    Step 10: 0.9^10 = 0.3487
    Step 20: 0.9^20 = 0.1216
    Step 50: 0.9^50 = 0.0052   <-- almost zero!

  After 50 steps, the original signal retains only 0.5% of its strength.

  If w = 1.1 (slightly more than 1):

    Step  1: 1.1^1  = 1.1000
    Step  5: 1.1^5  = 1.6105
    Step 10: 1.1^10 = 2.5937
    Step 20: 1.1^20 = 6.7275
    Step 50: 1.1^50 = 117.39   <-- exploded!


RNN vs ATTENTION: HOW THEY ACCESS EARLY TOKENS
================================================

  RNN (sequential access):

    Token 1 ---> Token 2 ---> Token 3 ---> ... ---> Token 50
        h_1 -------> h_2 -------> h_3 -------> ... -------> h_50

    To use info from Token 1 at Token 50:
    Signal must survive 49 multiplicative transformations.
    Path length: 49 steps.  Signal decay: w^49


  Attention (direct access):

    Token 1 -----------------------------------------------+
    Token 2 -------------------------------------------+   |
    Token 3 ---------------------------------------+   |   |
    ...                                            |   |   |
    Token 50 <-------------------------------------+---+---+

    To use info from Token 1 at Token 50:
    Direct connection. No intermediate transformations.
    Path length: 1 step.  No signal decay.

  This is why attention solves the long-range dependency problem.
```

---

## Hands-on Exploration

1. Take a calculator (or Python) and compute 0.95^n for n = 1, 5, 10, 20, 50, 100, 200. At what point does the value drop below 0.01 (1% of the original signal)? Now repeat with 0.99 -- even "almost 1" still decays. What does this tell you about an RNN's ability to remember the first word of a 200-word paragraph?

2. Consider the sentence: "The students who attended the lecture that the professor who won the award gave last semester passed the exam." The subject "students" and the verb "passed" are 15 words apart, separated by two nested relative clauses. Trace how an RNN would process this. At which step does it encounter "students"? At which step does it encounter "passed"? How many state updates separate them? If each update multiplies the hidden state by 0.9, how much of the "students" signal remains when "passed" is processed?

3. Design a sentence where the meaning of the last word depends critically on the first word, with at least 10 words in between. For example: "The key that was hidden under the third rock from the left opens [what?]." Now consider: would a BoW model, an n-gram model, or an RNN handle this best? Which would fail? Would attention help?

---

## Live Code

```python
# --- Sequence Modeling Challenges ---
# Only Python stdlib -- no external packages

import math
import random

random.seed(42)


# ============================================================
print("=" * 65)
print("PART 1: A SIMPLE RNN-LIKE COMPUTATION")
print("=" * 65)
print()
print("An RNN updates a hidden state at each step:")
print("  h_t = tanh(W_h * h_{t-1} + W_x * x_t)")
print()
print("We will simulate this with scalar values for clarity.")
print()


def tanh(x):
    """Hyperbolic tangent, clamped for numerical stability."""
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e_pos = math.exp(x)
    e_neg = math.exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)


# Assign each word a simple scalar "embedding"
word_values = {
    "the": 0.1, "cat": 0.8, "sat": 0.5, "on": 0.2,
    "mat": 0.7, "and": 0.05, "watched": 0.6, "birds": 0.9,
    "dog": 0.75, "chased": 0.65, "mouse": 0.85,
}

sentence = "the cat sat on the mat and watched the birds"
tokens = sentence.split()

# Simulate RNN with scalar hidden state
w_h = 0.7    # weight for previous hidden state
w_x = 0.5    # weight for input
h = 0.0      # initial hidden state

print(f"Sentence: \"{sentence}\"")
print(f"Weights: W_h = {w_h}, W_x = {w_x}")
print(f"Initial state: h_0 = {h}")
print()
print(f"  {'Step':>4}  {'Token':>10}  {'x_t':>6}  {'W_h*h + W_x*x':>16}  {'h_t (tanh)':>12}")
print(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*16}  {'-'*12}")

states = [h]
for t, token in enumerate(tokens):
    x_t = word_values.get(token, 0.3)
    pre_activation = w_h * h + w_x * x_t
    h = tanh(pre_activation)
    states.append(h)
    print(f"  {t+1:>4}  {token:>10}  {x_t:>6.2f}  {pre_activation:>16.4f}  {h:>12.4f}")

print()
print(f"Final hidden state: {h:.4f}")
print(f"This single number supposedly encodes the entire sentence.")
print()

# ============================================================
print("=" * 65)
print("PART 2: THE VANISHING SIGNAL -- REPEATED MULTIPLICATION")
print("=" * 65)
print()
print("Core problem: when we multiply by a factor < 1 repeatedly,")
print("signals from early steps shrink toward zero.")
print()

weights = [0.5, 0.7, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1]
steps = [1, 5, 10, 20, 50, 100]

print(f"  {'Weight':>8}", end="")
for s in steps:
    print(f"  {'Step '+str(s):>10}", end="")
print()
print(f"  {'-'*8}", end="")
for _ in steps:
    print(f"  {'-'*10}", end="")
print()

for w in weights:
    print(f"  {w:>8.2f}", end="")
    for s in steps:
        val = w ** s
        if abs(val) > 999999:
            print(f"  {'EXPLODED':>10}", end="")
        elif abs(val) < 0.0001:
            print(f"  {'~0.0000':>10}", end="")
        else:
            print(f"  {val:>10.4f}", end="")
    print()

print()
print("Observations:")
print("  - w < 1.0: signal VANISHES (shrinks to zero)")
print("  - w = 1.0: signal preserved perfectly (but this is unstable)")
print("  - w > 1.0: signal EXPLODES (grows without bound)")
print("  - There is NO safe fixed multiplier for long sequences.")
print()

# ============================================================
print("=" * 65)
print("PART 3: MEASURING INFORMATION RETENTION IN OUR RNN")
print("=" * 65)
print()
print("How much does the first word influence the final hidden state?")
print("We test by changing the first word and seeing how much")
print("the final state changes.")
print()


def run_rnn(tokens, word_vals, w_h, w_x):
    """Run a simple scalar RNN and return all hidden states."""
    h = 0.0
    states = [h]
    for token in tokens:
        x_t = word_vals.get(token, 0.3)
        h = tanh(w_h * h + w_x * x_t)
        states.append(h)
    return states


# Original sentence
original = "the cat sat on the mat and watched the birds"
original_tokens = original.split()
original_states = run_rnn(original_tokens, word_values, w_h, w_x)
original_final = original_states[-1]

# Change the first word to different words and measure impact
test_words = ["cat", "dog", "mouse", "birds", "mat"]

print(f"Original sentence: \"{original}\"")
print(f"Original final state: {original_final:.6f}")
print()

# Test changing word at different positions
positions_to_test = [0, 1, 4, 8, 9]  # first, second, middle, late, last

print(f"{'Position':>10}  {'Original':>10}  {'Changed to':>12}  "
      f"{'New final':>10}  {'Abs diff':>10}  {'Influence':>10}")
print(f"{'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

for pos in positions_to_test:
    original_word = original_tokens[pos]
    # Pick a replacement word that is different
    replacement = "mouse" if original_word != "mouse" else "dog"

    modified_tokens = original_tokens.copy()
    modified_tokens[pos] = replacement
    modified_states = run_rnn(modified_tokens, word_values, w_h, w_x)
    modified_final = modified_states[-1]

    diff = abs(original_final - modified_final)

    print(f"{pos:>10}  {original_word:>10}  {replacement:>12}  "
          f"{modified_final:>10.6f}  {diff:>10.6f}  ", end="")

    # Visual bar for influence
    bar_len = int(diff * 200)
    bar = "#" * min(bar_len, 40)
    print(bar)

print()
print("Notice: changing EARLY words has LESS impact on the final state")
print("than changing LATE words. The RNN forgets the beginning!")
print()

# ============================================================
print("=" * 65)
print("PART 4: LONG-RANGE DEPENDENCIES")
print("=" * 65)
print()
print("Language often requires connecting words far apart.")
print("We will show that our RNN loses this connection.")
print()

# Sentences where the first word determines the correct ending
examples = [
    {
        "prefix": "The dogs",
        "middle": "that the man who lived in the big house near the old church",
        "endings": ["bark loudly", "barks loudly"],
        "correct": 0,
        "explanation": "plural 'dogs' requires 'bark' (no -s), not 'barks'"
    },
    {
        "prefix": "The key",
        "middle": "which was hidden under the third rock from the left by the garden",
        "endings": ["opens the door", "eat the cake"],
        "correct": 0,
        "explanation": "'key' should associate with 'opens', not 'eat'"
    },
    {
        "prefix": "The musician",
        "middle": "who performed at the concert in the park on Saturday evening",
        "endings": ["played beautifully", "dissolved quickly"],
        "correct": 0,
        "explanation": "'musician' should associate with 'played'"
    },
]

for ex in examples:
    full_correct = f"{ex['prefix']} {ex['middle']} {ex['endings'][ex['correct']]}"
    tokens_correct = full_correct.lower().split()

    # How far apart are the key words?
    first_word = ex["prefix"].lower().split()[1]  # The main noun
    last_word = ex["endings"][ex["correct"]].lower().split()[0]  # The main verb

    distance = len(tokens_correct) - 2  # approximate

    print(f"  Subject: \"{ex['prefix']}\"")
    print(f"  Filler:  \"{ex['middle']}\"")
    print(f"  Correct: \"{ex['endings'][ex['correct']]}\"")
    print(f"  Wrong:   \"{ex['endings'][1 - ex['correct']]}\"")
    print(f"  Key link: '{first_word}' <-> '{last_word}' ({distance} words apart)")
    print(f"  Why: {ex['explanation']}")

    # Show signal decay
    decay_09 = 0.9 ** distance
    decay_095 = 0.95 ** distance

    print(f"  Signal retention (w=0.90): {decay_09:.4f} ({decay_09*100:.1f}%)")
    print(f"  Signal retention (w=0.95): {decay_095:.4f} ({decay_095*100:.1f}%)")
    print()

print("The farther apart the related words, the weaker the connection.")
print("RNNs struggle precisely because these dependencies are common.")
print()

# ============================================================
print("=" * 65)
print("PART 5: VISUALIZING STATE DECAY ACROSS A LONG SEQUENCE")
print("=" * 65)
print()

# Generate a longer sequence and track how a specific initial signal decays
sequence_length = 40

# Start with a strong initial signal
initial_signal = 1.0
decay_factor = 0.85

print(f"Tracking how a signal from position 0 decays over {sequence_length} steps")
print(f"Decay factor per step: {decay_factor}")
print()

signal = initial_signal
decay_positions = []
decay_signals = []
print(f"  {'Position':>8}  {'Signal':>8}  {'Remaining':>10}  Visualization")
print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*40}")

for pos in range(sequence_length + 1):
    pct = (signal / initial_signal) * 100
    bar_len = int(pct / 2.5)
    bar = "#" * max(bar_len, 0)

    decay_positions.append(pos)
    decay_signals.append(round(signal, 4))

    if pos % 4 == 0 or pos <= 3 or pos == sequence_length:
        print(f"  {pos:>8}  {signal:>8.4f}  {pct:>9.1f}%  |{bar}")

    signal *= decay_factor

print()

# Show at what position different thresholds are crossed
thresholds = [0.5, 0.1, 0.01, 0.001]
print("Signal drops below threshold at:")
for thresh in thresholds:
    if decay_factor <= 0 or decay_factor >= 1:
        continue
    steps_needed = math.log(thresh) / math.log(decay_factor)
    print(f"  {thresh*100:>6.1f}% signal remaining at step {steps_needed:>6.1f}")

print()

# --- Visualization: Signal Decay Over Sequence Length ---
show_chart({
    "type": "line",
    "title": f"Signal Decay Over Sequence (decay factor = {decay_factor})",
    "labels": [str(p) for p in decay_positions],
    "datasets": [{"label": "Signal Strength", "data": decay_signals, "color": "#ef4444"}],
    "options": {"x_label": "Sequence Position", "y_label": "Signal Strength"}
})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(decay_positions, decay_signals, color="#ef4444", linewidth=2)
    ax.fill_between(decay_positions, decay_signals, alpha=0.15, color="#ef4444")
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Signal Strength")
    ax.set_title(f"Signal Decay Over Sequence (decay factor = {decay_factor})")
    ax.axhline(y=0.5, color="#f59e0b", linestyle="--", linewidth=1, label="50% threshold")
    ax.axhline(y=0.1, color="#3b82f6", linestyle="--", linewidth=1, label="10% threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("signal_decay.png", dpi=100)
    plt.close()
    print("  [Saved matplotlib chart: signal_decay.png]")
except ImportError:
    pass

# ============================================================
print("=" * 65)
print("PART 6: WHY ATTENTION SOLVES THIS")
print("=" * 65)
print()
print("The fundamental problem:")
print("  RNN path from token 1 to token N: N-1 sequential steps")
print("  Each step multiplies the signal by a factor ~ w")
print("  After N steps, signal strength ~ w^(N-1)")
print()
print("The attention solution:")
print("  Attention path from token 1 to token N: 1 direct step")
print("  No intermediate multiplications")
print("  Signal strength: preserved fully")
print()

# Compare path lengths
print(f"  {'Seq Length':>12}  {'RNN Path':>10}  {'RNN Signal':>12}  "
      f"{'Attn Path':>10}  {'Attn Signal':>12}")
print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*12}")

rnn_vs_attn_labels = []
rnn_vs_attn_rnn = []
rnn_vs_attn_attn = []

for n in [5, 10, 20, 50, 100, 200, 500]:
    rnn_path = n - 1
    rnn_signal = 0.95 ** rnn_path
    attn_path = 1
    attn_signal = 1.0

    rnn_vs_attn_labels.append(str(n))
    rnn_vs_attn_rnn.append(round(rnn_signal, 6))
    rnn_vs_attn_attn.append(round(attn_signal, 6))

    rnn_str = f"{rnn_signal:.6f}" if rnn_signal > 0.000001 else "~0.000000"

    print(f"  {n:>12}  {rnn_path:>10}  {rnn_str:>12}  "
          f"{attn_path:>10}  {attn_signal:>12.6f}")

print()
print("At sequence length 100: RNN retains 0.6% of the original signal.")
print("At sequence length 500: RNN retains effectively 0% of the signal.")
print("Attention retains 100% at ANY distance.")

# --- Visualization: RNN vs Attention Signal Retention ---
show_chart({
    "type": "line",
    "title": "RNN vs Attention: Signal Retention by Sequence Length",
    "labels": rnn_vs_attn_labels,
    "datasets": [
        {"label": "RNN (w=0.95)", "data": rnn_vs_attn_rnn, "color": "#ef4444"},
        {"label": "Attention", "data": rnn_vs_attn_attn, "color": "#10b981"},
    ],
    "options": {"x_label": "Sequence Length", "y_label": "Signal Strength"}
})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rnn_vs_attn_labels, rnn_vs_attn_rnn, "o-", color="#ef4444", linewidth=2, label="RNN (w=0.95)")
    ax.plot(rnn_vs_attn_labels, rnn_vs_attn_attn, "o-", color="#10b981", linewidth=2, label="Attention")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Signal Strength")
    ax.set_title("RNN vs Attention: Signal Retention by Sequence Length")
    ax.legend()
    plt.tight_layout()
    plt.savefig("rnn_vs_attention.png", dpi=100)
    plt.close()
    print("  [Saved matplotlib chart: rnn_vs_attention.png]")
except ImportError:
    pass
print()
print("This is why transformers replaced RNNs for most NLP tasks.")
print("Attention provides direct connections between all positions,")
print("eliminating the long-range dependency problem entirely.")
print()

# ============================================================
print("=" * 65)
print("SUMMARY: THE PATH FROM BOW TO ATTENTION")
print("=" * 65)
print()
print("  BoW/TF-IDF:  No order.     No context.   No sequence.")
print("  N-grams:     Local order.  Local context. No long-range.")
print("  RNNs:        Full order.   Sequential.    Vanishing signal.")
print("  Attention:   Full order.   Direct access. No decay.")
print()
print("Each method solves the previous method's biggest limitation.")
print("Attention (and Transformers) solve the vanishing signal problem")
print("by letting every token directly attend to every other token.")
print("This is the subject of Phase 9.")
```

---

## Key Takeaways

- **Sequential processing matters because language builds meaning over time.** Each word modifies, extends, or reverses what came before. A model that reads word by word and maintains a running state can in principle capture these dynamics -- but only if the state faithfully preserves earlier information.
- **Repeated multiplication causes signals to vanish or explode.** An RNN updates its hidden state by multiplying by weight matrices at each step. After many steps, early signals are multiplied by the weight factor raised to a high power. If the factor is less than 1, the signal vanishes; if greater than 1, it explodes. There is no stable fixed multiplier for arbitrary-length sequences.
- **Long-range dependencies are pervasive in natural language.** Subjects and verbs, pronouns and antecedents, negation and the words it scopes over -- these relationships routinely span 10, 20, or 50+ words. RNNs struggle precisely because these common patterns require remembering information across many sequential steps.
- **Attention eliminates the long path problem.** Instead of routing information through a chain of N-1 intermediate states, attention mechanisms create direct connections between any two positions. The path length drops from N-1 to 1, and signals are preserved regardless of sequence length. This is the core motivation for the Transformer architecture explored in Phase 9.

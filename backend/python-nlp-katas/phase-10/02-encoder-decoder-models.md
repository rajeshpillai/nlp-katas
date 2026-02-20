# Encoder-Only vs Decoder-Only Models

> Phase 10 — Modern NLP Pipelines (Awareness) | Kata 10.2

---

## Concept & Intuition

### What problem are we solving?

Not all NLP tasks are the same. Some tasks require **understanding an entire input** (classify this email, find entities in this sentence, measure similarity between two documents). Other tasks require **generating new text** (complete this sentence, write the next paragraph, translate to another language). These two families of tasks have fundamentally different computational requirements.

Modern transformer-based architectures split along this line. **Encoder-only models** (BERT-style) process the full input bidirectionally -- every word can attend to every other word. **Decoder-only models** (GPT-style) process text left-to-right -- each word can only see what came before it. This is not an arbitrary design choice. It reflects what each architecture is optimized to do.

### Why earlier methods fail

**Classical methods** (BoW, TF-IDF) have no notion of directionality at all. They treat documents as unordered bags of features. This works for simple classification but cannot generate coherent text or model sequential dependencies.

**Simple sequence models** (basic RNNs) process text in one direction but struggle with long-range dependencies. They also cannot be efficiently parallelized during training, making them impractical at scale.

**A single architecture for all tasks** forces compromises. If you give the model access to the full input (bidirectional), it cannot generate text autoregressively because it would "see the future." If you restrict it to left-to-right processing, it loses information from the right context when doing classification. The split into encoder-only and decoder-only architectures resolves this tension.

### Mental models

- **Encoder-only (BERT-style) is like reading a complete document.** You see all the words at once and form a comprehensive understanding. You can answer questions about the text, classify its topic, find named entities -- anything that requires analyzing the full input.
- **Decoder-only (GPT-style) is like writing a sentence word by word.** You have written "The cat sat on the" and must predict the next word. You can only use what came before. This constraint is what makes text generation possible -- each prediction extends the sequence.
- **Masked language modeling** (encoder-only training) is like a fill-in-the-blank test. Given "The ___ sat on the mat", predict the masked word using context from both sides. This forces the model to build rich bidirectional representations.
- **Autoregressive modeling** (decoder-only training) is like finishing someone's sentence. Given "The cat sat on the", predict "mat". The model learns to generate text that follows naturally from what came before.

### Visual explanations

```
ENCODER-ONLY (BERT-style): BIDIRECTIONAL ATTENTION
====================================================

  Input:  "The  cat  [MASK]  on  the  mat"

  Each word attends to EVERY other word:

    The  ──→ cat, [MASK], on, the, mat
    cat  ──→ The, [MASK], on, the, mat
  [MASK] ──→ The, cat, on, the, mat         ← uses BOTH sides
    on   ──→ The, cat, [MASK], the, mat
    the  ──→ The, cat, [MASK], on, mat
    mat  ──→ The, cat, [MASK], on, the

  Prediction for [MASK]: "sat"
  (uses left context "The cat" AND right context "on the mat")

  Good for: classification, NER, similarity, question answering
  Bad for:  text generation (cannot mask the future during generation)


DECODER-ONLY (GPT-style): LEFT-TO-RIGHT ATTENTION
====================================================

  Input:  "The  cat  sat  on  the  ___"

  Each word can ONLY attend to previous words:

    The  ──→ (nothing -- first word)
    cat  ──→ The
    sat  ──→ The, cat
    on   ──→ The, cat, sat
    the  ──→ The, cat, sat, on
    ___  ──→ The, cat, sat, on, the        ← only left context

  Prediction for ___: "mat"
  (uses only preceding context -- cannot peek ahead)

  Good for: text generation, completion, conversation
  Bad for:  tasks needing full bidirectional context


TASK ALIGNMENT
===============

  Task                    Best architecture    Why
  ─────────────────────   ──────────────────   ─────────────────────
  Text classification     Encoder-only         Needs full input
  Named entity recog.     Encoder-only         Needs surrounding context
  Sentence similarity     Encoder-only         Needs to represent whole input
  Sentiment analysis      Encoder-only         Needs full sentence meaning

  Text completion         Decoder-only         Generates word by word
  Story generation        Decoder-only         Extends text sequentially
  Code completion         Decoder-only         Predicts next tokens
  Conversation            Decoder-only         Generates responses

  Translation             Encoder-Decoder      Reads source, generates target
  Summarization           Encoder-Decoder      Reads document, generates summary
```

---

## Hands-on Exploration

1. The code below simulates both masked language modeling (encoder-style) and autoregressive generation (decoder-style) using simple co-occurrence statistics. Run it and observe: when predicting a masked word, having context from BOTH sides produces better predictions than using only the left side. Why does bidirectional context help for fill-in-the-blank but create problems for generation?

2. Look at the autoregressive generation output. The model generates text one word at a time, each time choosing from words that commonly follow the current context. Notice how the generated text stays somewhat coherent locally but may drift topically. This is a fundamental property of left-to-right generation. What would happen if the context window were larger? Smaller?

3. Consider a real-world task: extracting product names from customer support emails. Would you want an encoder-only or decoder-only architecture? Now consider a different task: generating auto-reply suggestions for those same emails. How does the task requirement change which architecture fits better? Map three tasks from your own work to the correct architecture family.

---

## Live Code

```python
# --- Encoder-Only vs Decoder-Only: Simulation with Co-occurrence ---
# Pure Python stdlib -- no external dependencies

import math
import random
from collections import Counter, defaultdict

random.seed(42)


# ============================================================
# BUILD A LANGUAGE MODEL FROM CORPUS
# ============================================================

print("=" * 65)
print("BUILDING LANGUAGE MODEL FROM CORPUS")
print("=" * 65)

# A corpus with enough structure to demonstrate both approaches
corpus = [
    "the cat sat on the warm mat near the window",
    "the dog sat on the soft rug near the door",
    "the cat chased the small mouse across the yard",
    "the dog fetched the red ball across the park",
    "the cat slept on the warm blanket all morning",
    "the dog played on the green grass all afternoon",
    "the bird sang on the tall tree near the pond",
    "the fish swam in the clear pond near the rocks",
    "the cat watched the bird from the window ledge",
    "the dog watched the cat from the garden fence",
    "a warm breeze blew across the green yard today",
    "the small bird flew across the clear blue sky",
    "the red ball rolled across the soft green grass",
    "the tall tree stood near the old garden fence",
]

print(f"\nCorpus: {len(corpus)} sentences")


def tokenize(text):
    return text.lower().split()


# Build bigram model (word pairs) for autoregressive prediction
# Build context-window model for masked prediction
bigram_counts = defaultdict(Counter)      # bigram_counts[w1][w2] = count
context_counts = defaultdict(Counter)     # context_counts[context_word][target] = count
word_freq = Counter()

WINDOW = 2  # context window for masked prediction

for sentence in corpus:
    tokens = tokenize(sentence)
    for i, token in enumerate(tokens):
        word_freq[token] += 1

        # Bigram: current word predicts next word
        if i < len(tokens) - 1:
            bigram_counts[token][tokens[i + 1]] += 1

        # Context window: surrounding words predict center word
        for j in range(max(0, i - WINDOW), min(len(tokens), i + WINDOW + 1)):
            if i != j:
                context_counts[tokens[j]][token] += 1

vocab = sorted(word_freq.keys())
print(f"Vocabulary: {len(vocab)} words")
print(f"Total tokens: {sum(word_freq.values())}")


# ============================================================
# ENCODER-STYLE: MASKED LANGUAGE MODEL (BIDIRECTIONAL)
# ============================================================

print()
print("=" * 65)
print("ENCODER-STYLE: MASKED LANGUAGE MODEL (BIDIRECTIONAL)")
print("=" * 65)
print()
print("Task: predict the [MASK]ed word using context from BOTH sides.")
print("This simulates how encoder-only models like BERT are trained.\n")


def predict_masked_bidirectional(tokens, mask_pos, context_counts, top_n=5):
    """
    Predict a masked word using context from both sides.
    Collects evidence from all visible context words.
    """
    candidate_scores = Counter()

    for i, token in enumerate(tokens):
        if i == mask_pos:
            continue  # skip the masked position
        # Every context word votes for which word should fill the mask
        if token in context_counts:
            for candidate, count in context_counts[token].items():
                candidate_scores[candidate] += count

    total = sum(candidate_scores.values())
    results = []
    for word, score in candidate_scores.most_common(top_n):
        prob = score / total if total > 0 else 0
        results.append((word, prob))
    return results


def predict_masked_left_only(tokens, mask_pos, context_counts, top_n=5):
    """
    Predict a masked word using ONLY left context.
    This shows what you lose without bidirectional attention.
    """
    candidate_scores = Counter()

    for i in range(mask_pos):  # only words to the LEFT
        token = tokens[i]
        if token in context_counts:
            for candidate, count in context_counts[token].items():
                candidate_scores[candidate] += count

    total = sum(candidate_scores.values())
    results = []
    for word, score in candidate_scores.most_common(top_n):
        prob = score / total if total > 0 else 0
        results.append((word, prob))
    return results


# Test masked prediction on several examples
mask_tests = [
    ("the cat [MASK] on the warm mat", 2, "sat"),
    ("the [MASK] chased the small mouse", 1, "cat"),
    ("the dog played on the [MASK] grass", 5, "green"),
    ("the bird sang on the tall [MASK]", 6, "tree"),
    ("the cat watched the [MASK] from the window", 4, "bird"),
]

for sentence, mask_pos, true_word in mask_tests:
    tokens = tokenize(sentence.replace("[MASK]", "[mask]"))
    tokens[mask_pos] = "[mask]"

    print(f'  Sentence: "{sentence}"')
    print(f'  True word: "{true_word}"')

    # Bidirectional prediction (encoder-style)
    bi_preds = predict_masked_bidirectional(tokens, mask_pos,
                                            context_counts, top_n=5)
    bi_str = ", ".join(f'{w} ({p:.3f})' for w, p in bi_preds)
    bi_rank = next((i + 1 for i, (w, _) in enumerate(bi_preds)
                    if w == true_word), ">5")

    # Left-only prediction (limited, like decoder seeing only past)
    left_preds = predict_masked_left_only(tokens, mask_pos,
                                          context_counts, top_n=5)
    left_str = ", ".join(f'{w} ({p:.3f})' for w, p in left_preds)
    left_rank = next((i + 1 for i, (w, _) in enumerate(left_preds)
                      if w == true_word), ">5")

    print(f'  Bidirectional: [{bi_str}]  (true word rank: {bi_rank})')
    print(f'  Left-only:     [{left_str}]  (true word rank: {left_rank})')
    print()

print("  Bidirectional context (encoder-style) produces better predictions")
print("  because it uses evidence from BOTH sides of the masked word.")
print("  Left-only context misses crucial information from the right.")


# ============================================================
# DECODER-STYLE: AUTOREGRESSIVE GENERATION (LEFT-TO-RIGHT)
# ============================================================

print()
print("=" * 65)
print("DECODER-STYLE: AUTOREGRESSIVE GENERATION (LEFT-TO-RIGHT)")
print("=" * 65)
print()
print("Task: generate text by predicting one word at a time.")
print("Each word is chosen based ONLY on preceding words.")
print("This simulates how decoder-only models like GPT generate text.\n")


def generate_autoregressive(seed_tokens, bigram_counts, max_length=10,
                            temperature=1.0):
    """
    Generate text left-to-right, one word at a time.
    Each new word depends only on the previous word (bigram model).
    """
    tokens = list(seed_tokens)
    history = []  # track predictions for display

    for step in range(max_length):
        last_word = tokens[-1]
        if last_word not in bigram_counts:
            break

        candidates = bigram_counts[last_word]
        if not candidates:
            break

        # Compute probabilities
        total = sum(candidates.values())
        probs = [(w, c / total) for w, c in candidates.items()]
        probs.sort(key=lambda x: x[1], reverse=True)

        # Sample with temperature
        if temperature <= 0.01:
            # Greedy: pick the most likely word
            chosen = probs[0][0]
        else:
            # Weighted random sampling
            weights = [p ** (1.0 / temperature) for _, p in probs]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            r = random.random()
            cumulative = 0
            chosen = probs[0][0]
            for (word, _), w in zip(probs, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = word
                    break

        history.append((last_word, probs[:3], chosen))
        tokens.append(chosen)

    return tokens, history


# Generate from several prompts
prompts = [
    ["the", "cat"],
    ["the", "dog"],
    ["the", "bird"],
    ["a", "warm"],
]

for prompt in prompts:
    prompt_str = " ".join(prompt)
    print(f'  Prompt: "{prompt_str}"')

    # Greedy generation
    tokens_greedy, history_greedy = generate_autoregressive(
        prompt, bigram_counts, max_length=8, temperature=0.01)
    print(f'  Greedy:  "{" ".join(tokens_greedy)}"')

    # Sampled generation (more variety)
    random.seed(42)
    tokens_sampled, history_sampled = generate_autoregressive(
        prompt, bigram_counts, max_length=8, temperature=0.8)
    print(f'  Sampled: "{" ".join(tokens_sampled)}"')

    print()

# Show the step-by-step generation process for one example
print("--- Step-by-Step Generation (decoder-style) ---\n")
prompt = ["the", "cat"]
random.seed(42)
tokens, history = generate_autoregressive(
    prompt, bigram_counts, max_length=6, temperature=0.01)

print(f'  Starting with: "{" ".join(prompt)}"')
print()
for step, (last_word, top_preds, chosen) in enumerate(history):
    context_so_far = " ".join(prompt + [h[2] for h in history[:step]])
    pred_str = ", ".join(f'"{w}" ({p:.2f})' for w, p in top_preds)
    print(f'  Step {step + 1}: context = "{context_so_far}"')
    print(f'          candidates: [{pred_str}]')
    print(f'          chose: "{chosen}"')
    print()

print(f'  Final: "{" ".join(tokens)}"\n')
print("  Each word is predicted from ONLY the preceding context.")
print("  The model cannot look ahead -- this is the autoregressive constraint")
print("  that makes text generation possible.")


# ============================================================
# COMPARISON: SAME TEXT, DIFFERENT ARCHITECTURES
# ============================================================

print()
print("=" * 65)
print("COMPARISON: SAME TEXT, DIFFERENT TASKS")
print("=" * 65)

test_sentence = "the cat sat on the warm mat"
test_tokens = tokenize(test_sentence)

print(f'\n  Input: "{test_sentence}"\n')

# Encoder-style: classify / analyze the full input
print("  ENCODER-STYLE (analyze full input):")
print("  The model sees ALL words simultaneously.")
print("  It can answer: What is the subject? Where did it sit?\n")

# Show attention pattern: each word attends to all others
print("  Attention pattern (bidirectional):")
print(f"  {'':>8s}", end="")
for t in test_tokens:
    print(f"{t:>6s}", end="")
print()
for i, t in enumerate(test_tokens):
    print(f"  {t:>8s}", end="")
    for j in range(len(test_tokens)):
        print(f"{'  *  ' :>6s}", end="")  # all positions attend to all
    print()

print()

# Decoder-style: generate continuations
print("  DECODER-STYLE (generate from prefix):")
print("  The model sees words LEFT-TO-RIGHT only.")
print("  It can answer: What word comes next?\n")

# Show causal attention pattern
print("  Attention pattern (causal / left-to-right):")
print(f"  {'':>8s}", end="")
for t in test_tokens:
    print(f"{t:>6s}", end="")
print()
for i, t in enumerate(test_tokens):
    print(f"  {t:>8s}", end="")
    for j in range(len(test_tokens)):
        if j <= i:
            print(f"{'  *  ':>6s}", end="")  # can attend
        else:
            print(f"{'  -  ':>6s}", end="")  # masked (cannot see future)
    print()

print()
print("  *  = can attend     -  = masked (cannot see)")
print()
print("  Encoder sees the full triangle: every word connects to every word.")
print("  Decoder sees the lower triangle: each word only connects to the past.")


# ============================================================
# WHEN TO USE WHICH ARCHITECTURE
# ============================================================

print()
print("=" * 65)
print("DECISION GUIDE: WHICH ARCHITECTURE FOR WHICH TASK?")
print("=" * 65)

tasks = [
    ("Classify email as spam/not spam", "Encoder-only",
     "Needs to analyze the full email content"),
    ("Extract person names from text", "Encoder-only",
     "Each token needs context from both sides"),
    ("Measure similarity of two docs", "Encoder-only",
     "Must represent complete documents"),
    ("Complete a code snippet", "Decoder-only",
     "Must generate tokens left-to-right"),
    ("Write a product description", "Decoder-only",
     "Must generate coherent text from a prompt"),
    ("Answer questions in a chatbot", "Decoder-only",
     "Must generate response text sequentially"),
    ("Translate English to French", "Encoder-Decoder",
     "Must read full source, then generate target"),
    ("Summarize a long article", "Encoder-Decoder",
     "Must comprehend input, then generate summary"),
]

print()
print(f"  {'Task':<38s} {'Architecture':<18s} {'Reason'}")
print(f"  {'-'*38} {'-'*18} {'-'*40}")
for task, arch, reason in tasks:
    print(f"  {task:<38s} {arch:<18s} {reason}")

print()
print("  The choice is not about which is \"better\" -- it is about")
print("  which computational structure matches the task requirements.")
print()
print("  Encoder-only: compute rich representations of existing text.")
print("  Decoder-only: compute the next token given preceding tokens.")
print("  These are different computations, suited to different problems.")
```

---

## Key Takeaways

- **Encoder-only models compute bidirectional representations.** Every token attends to every other token, producing representations that capture the full context of the input. This makes them well-suited for classification, entity recognition, and similarity tasks where the model must analyze a complete input.
- **Decoder-only models compute left-to-right predictions.** Each token can only attend to previous tokens, which is the constraint that makes sequential text generation possible. The model predicts one token at a time, extending the sequence autoregressively.
- **The architecture choice is driven by the task, not by preference.** Classification needs bidirectional context (encoder). Generation needs autoregressive prediction (decoder). Translation needs both (encoder-decoder). These are engineering decisions based on computational requirements.
- **Neither architecture "understands" language.** Both compute statistical patterns over token sequences. The encoder computes patterns using full bidirectional context. The decoder computes patterns using left-to-right context. The difference is structural -- what information flows where -- not a difference in comprehension.

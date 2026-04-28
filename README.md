# 🐾 DogTalk AI

**Understanding Dog Emotions Through Bark Analysis**  
*CS 5588 — Week 14 · Foundation Models for Speech, Music, and Sound AI*

---

## Overview

DogTalk AI is a two-pipeline system that bridges the emotional gap between dogs and humans using state-of-the-art audio AI models.

| Pipeline | Direction | Description |
|---|---|---|
| **Pipeline A** | 🐶 → 👤 Dog → Human | Analyzes dog bark audio and classifies it into one of 4 emotions |
| **Pipeline B** | 👤 → 🐶 Human → Dog | Takes a human emotion as text and generates a corresponding dog bark audio clip |

**Emotion Classes:** `happy 😄` · `anxious 😰` · `alert ⚡` · `bored 😴`

---

## Project Structure

```
dogtalk-ai/
├── pipeline1.ipynb              # Pipeline A: Bark → Emotion classifier
├── pipeline2.ipynb              # Pipeline B: Emotion → Bark generator
├── README.md                    # This file
├── INSTRUCTIONS.md              # Setup and run guide
├── LIBRARIES.md                 # All dependencies
├── sample_outputs/
│   ├── SAMPLE_OUTPUTS.md        # Written output logs & metrics
│   ├── dogtalk_spectrograms.png         # Waveform + spectrogram grid
│   ├── dogtalk_training_loss.png        # Training loss curve
│   ├── dogtalk_confusion_matrix.png     # Confusion matrix
│   ├── dogtalk_baseline_vs_engineered.png  # Prompt comparison chart
│   └── dogtalk_evaluation.csv           # Full evaluation table
```

---

## Architecture

### Pipeline A — Dog Bark → Emotion Label

```
🎙️ Raw Audio (.wav)
    └─► Wav2Vec2 Feature Extractor (facebook/wav2vec2-base-960h)
            └─► Mean-pooled hidden states (768-dim embedding)
                    └─► Fine-tuned Emotion Classifier (PyTorch MLP)
                            └─► 🏷️ Emotion Label + Confidence Scores
```

### Pipeline B — Human Emotion → Dog Bark Audio

```
📝 Human Emotion Text
    └─► Prompt Engineering (baseline vs. engineered prompts)
            └─► AudioLDM2 (cvssp/audioldm2)
                    └─► 🔊 Generated Bark Audio (.wav)
```

---

## Key Results

### Pipeline A
- **Test Accuracy:** 100%
- **Macro F1-Score:** 1.00
- Perfect precision and recall across all 4 emotion classes

### Pipeline B
| Metric | Baseline Prompt | Engineered Prompt |
|---|---|---|
| Prompt Alignment (1–5) | 2.1 | 4.2 |
| Realism (1–5) | 2.4 | 3.8 |
| Diversity (1–5) | 1.8 | 3.5 |

Engineered prompts produced more controlled spectral energy, tighter frequency bands, and lower amplitude variance.

---

## Models Used

| Model | Source | Role |
|---|---|---|
| `facebook/wav2vec2-base-960h` | Hugging Face | Audio feature extraction |
| `cvssp/audioldm2` | Hugging Face | Text-to-audio generation |
| Custom EmotionClassifier (MLP) | PyTorch | Bark emotion classification |

---

## Limitations

- Audio samples in Pipeline A are synthetically generated, not recorded from real dogs.
- Small test set (n=8); results are not yet generalizable to real-world deployment.
- Only 4 emotion classes supported.
- No real-time or mobile deployment — batch/notebook only.
- Pipeline B evaluated qualitatively (waveform/spectrogram), not quantitatively.

---

## AI Tools Disclosure

| Tool | Usage |
|---|---|
| OpenAI GPT-4 / Claude | Pipeline B LLM for bark sound generation |
| GitHub Copilot | Boilerplate code assistance |
| ChatGPT | Brainstorming and prompt design |
| Claude (Anthropic) | Documentation and slide writing |

---

## Course Context

**Course:** CS 5588  
**Theme:** Foundation Models for Speech, Music, and Sound AI  
**Project:** DogTalk AI — AI Capstone, 2025

# 🐶 DogTalk AI — Sample Outputs

All outputs below were produced by running `pipeline1.ipynb` and `pipeline2.ipynb`.  
Visual outputs (PNG charts) and audio files (.wav) are saved alongside the notebooks.

---

## Pipeline A Outputs

### Step 3 — Dataset Generation

```
✅ Dataset ready: 40 samples across 4 emotions
Emotions: happy 😄, anxious 😰, alert ⚡, bored 😴
```

### Step 4 — Visualizations

**Output file:** `dogtalk_spectrograms.png`

A 2×4 grid showing:
- **Row 1 (Waveforms):** Amplitude plots (first 500 samples) for each of the 4 emotions
  - `happy 😄` — short, high-pitched repeating bursts (~900 Hz)
  - `anxious 😰` — rapid mid-range whimper signal (~600 Hz)
  - `alert ⚡` — sharp single burst with exponential decay (~400 Hz)
  - `bored 😴` — low-amplitude slow modulation (~300 Hz)
- **Row 2 (Mel Spectrograms):** 64-bin mel spectrograms showing distinct frequency-energy patterns per emotion

### Step 5 — Embedding Extraction

```
Extracting embeddings...
✅ Embeddings shape: torch.Size([40, 768])
```

### Step 6 — Training

**Output file:** `dogtalk_training_loss.png`

Training loss curve over 50 epochs. Loss converges from ~1.38 (random baseline) to near 0.

```
✅ Training complete. Final loss: 0.0021
```

### Step 7 — Evaluation

**Output file:** `dogtalk_confusion_matrix.png`

```
==================================================
PIPELINE A — Classification Report
==================================================
              precision    recall  f1-score   support

       happy       1.00      1.00      1.00         2
     anxious       1.00      1.00      1.00         2
       alert       1.00      1.00      1.00         2
       bored       1.00      1.00      1.00         2

    accuracy                           1.00         8
   macro avg       1.00      1.00      1.00         8
weighted avg       1.00      1.00      1.00         8
```

**Confusion Matrix** (all values on diagonal — zero misclassifications):

|  | happy | anxious | alert | bored |
|---|---|---|---|---|
| **happy** | 2 | 0 | 0 | 0 |
| **anxious** | 0 | 2 | 0 | 0 |
| **alert** | 0 | 0 | 2 | 0 |
| **bored** | 0 | 0 | 0 | 2 |

### Step 8 — Real-time Inference Demo

Sample output for each emotion:

```
────────────────────────────
Testing with synthesized happy bark...

🎵 Bark audio (play below): [Audio widget]

🐶 DogTalk AI says: "happy" 😄
   Confidence: 99.3%

   All scores:
   😄 happy    ████████████████████  99.3%
   😰 anxious                         0.4%
   ⚡ alert                           0.2%
   😴 bored                           0.1%

────────────────────────────
Testing with synthesized anxious bark...

🐶 DogTalk AI says: "anxious" 😰
   Confidence: 98.7%

────────────────────────────
Testing with synthesized alert bark...

🐶 DogTalk AI says: "alert" ⚡
   Confidence: 99.1%

────────────────────────────
Testing with synthesized bored bark...

🐶 DogTalk AI says: "bored" 😴
   Confidence: 97.8%
```

---

## Pipeline B Outputs

### Cell 6 — Generated Audio Files

8 audio clips generated (baseline + engineered per emotion):

| Filename | Emotion | Prompt Type |
|---|---|---|
| `dogtalk_happy_baseline.wav` | happy | Baseline |
| `dogtalk_happy_engineered.wav` | happy | Engineered |
| `dogtalk_anxious_baseline.wav` | anxious | Baseline |
| `dogtalk_anxious_engineered.wav` | anxious | Engineered |
| `dogtalk_alert_baseline.wav` | alert | Baseline |
| `dogtalk_alert_engineered.wav` | alert | Engineered |
| `dogtalk_bored_baseline.wav` | bored | Baseline |
| `dogtalk_bored_engineered.wav` | bored | Engineered |

### Prompt Comparison (happy emotion)

| | Baseline | Engineered |
|---|---|---|
| **Prompt** | `"dog barking"` | `"excited happy dog barking playfully, short high-pitched yips, energetic tail-wagging bark, golden retriever, clear recording"` |
| **Spectral energy** | Diffuse, spread across wide frequency range | Concentrated in 500–1500 Hz band |
| **Amplitude variance** | High | Low, consistent envelope |
| **Perceived realism** | Generic | More recognizably dog-like |

**Output file:** `dogtalk_baseline_vs_engineered.png`  
Shows side-by-side waveforms and mel spectrograms for baseline (red) vs engineered (green) prompts.

### Cell 8 — Human → Dog End-to-End Demo

```
==================================================
👤 Human: "I feel happy and excited!"
🐶 Dog would feel: happy 😄
[Audio: dogtalk_human_happy.wav]

==================================================
👤 Human: "I feel very stressed out."
🐶 Dog would feel: anxious 😰
[Audio: dogtalk_human_stressed.wav]

==================================================
👤 Human: "I am angry and alert."
🐶 Dog would feel: alert ⚡
[Audio: dogtalk_human_angry.wav]

==================================================
👤 Human: "I am tired and bored."
🐶 Dog would feel: bored 😴
[Audio: dogtalk_human_tired.wav]

🎉 Pipeline B complete! All .wav files saved.
```

### Cell 9 — Full Evaluation Summary Table

**Output file:** `dogtalk_evaluation.csv`

| Pipeline | Setting | Metric | Score |
|---|---|---|---|
| A — Bark to Emotion | Baseline (random) | Accuracy | 25.0% |
| A — Bark to Emotion | Fine-tuned classifier | Accuracy | 100% |
| B — Emotion to Bark | Baseline prompt | Prompt alignment (1–5) | 2.1 |
| B — Emotion to Bark | Baseline prompt | Realism (1–5) | 2.4 |
| B — Emotion to Bark | Baseline prompt | Diversity (1–5) | 1.8 |
| B — Emotion to Bark | Engineered prompt | Prompt alignment (1–5) | 4.2 |
| B — Emotion to Bark | Engineered prompt | Realism (1–5) | 3.8 |
| B — Emotion to Bark | Engineered prompt | Diversity (1–5) | 3.5 |

---

## Summary

| Metric | Value |
|---|---|
| Pipeline A Test Accuracy | **100%** |
| Pipeline A Macro F1 | **1.00** |
| Pipeline B Engineered Prompt Alignment | **4.2 / 5.0** |
| Pipeline B Realism Improvement (vs baseline) | **+58%** (2.4 → 3.8) |
| Total audio files generated | **12 `.wav` files** |
| Total visual outputs | **4 `.png` files** |

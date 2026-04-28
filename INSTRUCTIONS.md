# 🛠️ DogTalk AI — Setup & Run Instructions

## Requirements

- Python 3.9+
- Google Colab (recommended) or a local environment with GPU support
- ~8 GB RAM minimum; GPU strongly recommended for Pipeline B (AudioLDM2)

---

## Quick Start

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `pipeline1.ipynb` and `pipeline2.ipynb` from this repository
3. Set runtime to **GPU**: `Runtime → Change runtime type → T4 GPU`
4. Run each notebook top-to-bottom

> ⚠️ Pipeline B (`pipeline2.ipynb`) requires a **fresh Colab session** due to package version conflicts. Before running it, go to `Runtime → Disconnect and delete runtime` to start clean.

---

## Pipeline A — Bark → Emotion (`pipeline1.ipynb`)

### Step-by-step

1. **Open** `pipeline1.ipynb` in Colab
2. **Run Step 0** — installs all dependencies:
   ```
   transformers datasets torch torchaudio soundfile librosa
   matplotlib scikit-learn diffusers accelerate ipython
   ```
3. **Run Steps 1–8 in order**

| Step | What it does |
|---|---|
| Step 0 | Install dependencies |
| Step 1 | Imports, device check (CPU/GPU) |
| Step 2 | Load `wav2vec2-base-960h` feature extractor |
| Step 3 | Synthesize dog bark dataset (4 emotions × 10 samples each) |
| Step 4 | Visualize waveforms and mel spectrograms |
| Step 5 | Extract 768-dim wav2vec2 embeddings |
| Step 6 | Train emotion classifier (50 epochs) + plot training loss |
| Step 7 | Evaluate: classification report + confusion matrix |
| Step 8 | Real-time inference demo on each emotion |

**Expected runtime:** ~5–10 minutes on CPU; ~2–3 minutes on GPU

**Outputs generated:**
- `dogtalk_spectrograms.png`
- `dogtalk_training_loss.png`
- `dogtalk_confusion_matrix.png`

---

## Pipeline B — Emotion → Bark (`pipeline2.ipynb`)

### ⚠️ Important: Version Pinning Required

Pipeline B uses `AudioLDM2`, which requires specific package versions. Package conflicts will occur if this notebook is run in the same session as Pipeline A.

**Steps:**

1. Start a **fresh Colab session** (`Runtime → Disconnect and delete runtime → Connect`)
2. **Run Cell 1** to install pinned packages:
   ```
   transformers==4.40.2
   diffusers==0.27.2
   accelerate==0.30.1
   huggingface_hub==0.20.3
   safetensors
   peft==0.10.0
   soundfile librosa scipy
   ```
3. **Restart the session** after Cell 1: `Runtime → Restart session`
4. **Run Cells 2–9 in order** (do NOT re-run Cell 1)

| Cell | What it does |
|---|---|
| Cell 1 | Install pinned packages (run once, then restart) |
| Cell 2 | Verify versions + imports |
| Cell 3 | Load `AudioLDM2` model (~3 min download) |
| Cell 4 | Define baseline vs engineered prompts for all 4 emotions |
| Cell 5 | Define `generate_bark()` function |
| Cell 6 | Generate all 8 audio clips (baseline + engineered per emotion) |
| Cell 7 | Compare baseline vs engineered: waveform + spectrogram chart |
| Cell 8 | End-to-end Human → Dog demo |
| Cell 9 | Print and save evaluation summary table |

**Expected runtime:** ~20–30 minutes on GPU (AudioLDM2 is compute-heavy)

**Outputs generated:**
- `dogtalk_{emotion}_baseline.wav` × 4
- `dogtalk_{emotion}_engineered.wav` × 4
- `dogtalk_human_{emotion}.wav` × 4
- `dogtalk_baseline_vs_engineered.png`
- `dogtalk_evaluation.csv`

---

## Local Installation (Advanced)

If running locally instead of Colab:

```bash
# Create a virtual environment
python -m venv dogtalk-env
source dogtalk-env/bin/activate   # Windows: dogtalk-env\Scripts\activate

# Install Pipeline A dependencies
pip install transformers==4.38.2 datasets torch torchaudio soundfile \
            librosa matplotlib scikit-learn diffusers accelerate ipython

# Install Pipeline B dependencies (separate env recommended)
pip install transformers==4.40.2 diffusers==0.27.2 accelerate==0.30.1 \
            huggingface_hub==0.20.3 safetensors peft==0.10.0 \
            soundfile librosa scipy

# Launch Jupyter
jupyter notebook
```

> 💡 A CUDA-enabled GPU is strongly recommended for Pipeline B. CPU inference with AudioLDM2 can take 30+ minutes per clip.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ImportError` or version conflict in Pipeline B | Ensure you started a **fresh Colab session** and ran only Cell 1 before restarting |
| `CUDA out of memory` | Reduce `num_inference_steps` to 10 or lower `audio_length_in_s` |
| `wav2vec2` download fails | Check internet connection; Hugging Face Hub may be temporarily slow |
| AudioLDM2 takes too long | Use GPU runtime in Colab; CPU inference is not practical for this model |
| `peft` version error | Run `!pip uninstall -y peft && pip install peft==0.10.0` |

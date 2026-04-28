# 📦 DogTalk AI — Libraries Used

## Pipeline A (`pipeline1.ipynb`)

| Library | Version | Purpose |
|---|---|---|
| `torch` | latest | Deep learning framework; model training and inference |
| `torchaudio` | latest | Audio loading and preprocessing utilities |
| `transformers` | `4.38.2` | Load and run `wav2vec2-base-960h` for feature extraction |
| `datasets` | latest | Hugging Face dataset utilities |
| `librosa` | latest | Audio analysis, mel spectrogram computation |
| `soundfile` | latest | Reading and writing `.wav` audio files |
| `numpy` | latest | Numerical operations and array manipulation |
| `matplotlib` | latest | Plotting waveforms, spectrograms, and training curves |
| `scikit-learn` | latest | `classification_report` and `confusion_matrix` |
| `IPython` | latest | `Audio` and `display` for in-notebook audio playback |

### Install command
```bash
pip install transformers==4.38.2 datasets torch torchaudio soundfile \
            librosa matplotlib scikit-learn diffusers accelerate ipython
```

---

## Pipeline B (`pipeline2.ipynb`)

| Library | Version | Purpose |
|---|---|---|
| `torch` | latest | Tensor operations and GPU inference |
| `transformers` | `4.40.2` | Tokenizers and model components for AudioLDM2 |
| `diffusers` | `0.27.2` | `AudioLDM2Pipeline` for text-to-audio generation |
| `accelerate` | `0.30.1` | Efficient model loading and device management |
| `huggingface_hub` | `0.20.3` | Model download and caching |
| `safetensors` | latest | Safe model weight serialization |
| `peft` | `0.10.0` | Parameter-efficient fine-tuning support (required by AudioLDM2) |
| `soundfile` | latest | Writing generated audio to `.wav` files |
| `librosa` | `0.10.1` | Mel spectrogram generation for output visualization |
| `numpy` | `1.26.4` | Array operations |
| `scipy` | latest | Signal processing utilities |
| `matplotlib` | latest | Waveform and spectrogram comparison plots |
| `pandas` | latest | Evaluation summary table creation and CSV export |
| `IPython` | latest | In-notebook audio playback |

### Install command
```bash
# Run first, then restart session before importing
pip install transformers==4.40.2 diffusers==0.27.2 accelerate==0.30.1 \
            huggingface_hub==0.20.3 safetensors peft==0.10.0 \
            soundfile librosa==0.10.1 scipy numpy==1.26.4 pandas

# Fix numba compatibility
pip install numba==0.59.1
```

---

## Pre-trained Models (Hugging Face Hub)

| Model ID | Pipeline | Description |
|---|---|---|
| `facebook/wav2vec2-base-960h` | A | Wav2Vec2 model pretrained on 960 hours of LibriSpeech; used as audio feature extractor |
| `cvssp/audioldm2` | B | AudioLDM2 latent diffusion model for text-to-audio generation |

Both models are downloaded automatically on first run via the Hugging Face Hub. An internet connection is required.

---

## Custom Model (Trained In-Notebook)

| Component | Details |
|---|---|
| **Name** | `EmotionClassifier` |
| **Type** | PyTorch `nn.Module` — Multi-Layer Perceptron |
| **Input** | 768-dim wav2vec2 embedding |
| **Architecture** | `Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→64) → ReLU → Linear(64→4)` |
| **Output** | 4-class logits (happy / anxious / alert / bored) |
| **Optimizer** | Adam, LR=1e-3 |
| **Loss** | Cross-Entropy |
| **Epochs** | 50 |

---

## Version Compatibility Note

Pipeline A and Pipeline B use **incompatible versions** of `transformers` (4.38.2 vs 4.40.2) and `diffusers`. They must be run in **separate Colab sessions** or **separate virtual environments**. See `INSTRUCTIONS.md` for details.

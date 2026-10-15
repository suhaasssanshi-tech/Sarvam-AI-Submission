# Real-Time Code-Mixed Hindi-to-English Speech-to-Speech Translation
## Inter IIT Tech Meet 15.0 — Sarvam.ai Qualifying Mini Problem Statement
**Track:** Signal Processing  
**Institution:** Indian Institute of Technology Patna (`@iitp.ac.in`)  
**Hardware Testbed:** Intel Core i7-14650HX (16C/24T) + NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6)

---

## Deliverables & Repository Structure

- **Submission Report (HTML):** [`report_mini_ps.html`](report_mini_ps.html)- Final Report .html format
- **Compiled Submission Report (PDF):** [`sarvam_mini_ps.pdf`](sarvam_mini_ps.pdf)-Final Report .pdf format 
- **Extended Technical Monograph (PDF):** [`sarvam_extended_draft_16p.pdf`](sarvam_extended_draft_16p.pdf) — Complete 16-page extended reference draft with full system derivations and extended proofs.
- **Official Problem Statement:** [`sarvam_official_mini_ps.pdf`](sarvam_official_mini_ps.pdf) — Official Inter IIT Tech Meet 15.0 Sarvam.ai qualifying document.
- **Detailed Benchmark Results:** [`RESULTS.md`](RESULTS.md) — Live reproduction logs, SI-SDR metrics across 5 degraded conditions, and segmental ASR evaluations.

```
SARVAM/
├── report_mini_ps.html                # Official 4-page IEEE/ACL publication report
├── sarvam_mini_ps.pdf                 # Strict 4-page submission PDF (1.76 MB)
├── sarvam_extended_draft_16p.pdf      # Complete 16-page extended monograph
├── sarvam_official_mini_ps.pdf        # Official Inter IIT Tech Meet 15.0 Problem Statement
├── RESULTS.md                         # Quantitative benchmark logs & SI-SDR results
├── requirements.txt                   # Reproducible 1-step Python dependencies
├── verify_setup.py                    # Hardware & CUDA verification script
├── scripts/
│   ├── causal_frontend.py             # Strictly causal 80 ms audio front-end engine
│   ├── eval_part_e.py                 # SI-SDR benchmark & figure generation
│   ├── generate_degraded_clips.py     # Synthesizes 5 degraded audio test conditions
│   ├── generate_html_report.py        # Generates report_mini_ps.html
│   ├── inspect_pdf.py                 # Validates exact 4-page PDF constraint
│   ├── run_live_interview.py          # Unified CLI entrypoint
│   └── run_interview_pipeline.py      # End-to-end ASR, alignment, & entity verification
└── data/
    ├── audio/                         # Standardized 24 kHz mono reference clips (clip_1 to clip_6)
    ├── raw/                           # Original audio recordings (clip_1 to clip_6)
    ├── alignments/                    # Audacity sub-20ms word alignment labels
    ├── part_e_audio/                  # Degraded conditions (noisy 15/10/5 dB, telephone 8k)
    ├── part_e_enhanced/               # Enhanced audio outputs across all denoisers
    └── part_e_plots/                  # High-resolution benchmark figures (Fig E.1 & E.2)
```

---

## System Architecture & Key Components

### 1. Strictly Causal 80 ms Front-End (`scripts/causal_frontend.py`)
Live Hindi speech streams in chunks of **80 ms = 1,920 samples @ 24 kHz (3,840 bytes, `pcm_s16le`)**. In strict adherence to **Rule 6 (Zero Future Lookahead)**:
- **Causal Stateful Resampler:** Polyphase FIR filter maintaining carry-over state buffers to eliminate boundary phase distortion across chunk seams.
- **Classical Causal Wiener Filter:** 20 ms sub-frames (10 ms hop, 50% overlap-add) with Decision-Directed a-priori SNR estimation ($\alpha_{DD} = 0.96$) and causal VAD-gated noise PSD tracking.
- **Neural Causal Denoiser (`DeepFilterNet3`):** Two-stage ERB envelope suppression + Deep Filtering with causal state persistence on NVIDIA RTX 5060 GPU.
- **Causal Silence Trimming (`Silero VAD`):** 512-sample sub-windows @ 16 kHz with a **240 ms (3-frame) causal hangover counter** to bridge natural inter-phrase pauses without clipping speech onsets.
- **Smart Bandwidth & SNR Bypass Gate (`SmartCausalGate`):** Evaluates instantaneous frame SNR and high-band spectral energy ratio:
  $$R = \frac{\sum_{f=3.4\text{ kHz}}^{12\text{ kHz}} |X(f)|^2}{\sum_{f=0.3\text{ kHz}}^{3.4\text{ kHz}} |X(f)|^2}$$
  Automatically bypasses destructive spectral attenuation when clean PSTN telephone speech ($R < 0.0008$) or high-SNR speech ($>20\text{ dB}$) is detected, preserving critical consonant bursts.

### 2. High-Accuracy Forced Alignment & Universal Verification (`scripts/run_interview_pipeline.py`)
- **5 ms RMS Energy-Clamped MMS Aligner:** Multilingual Meta MMS CTC aligner clamped with short-time RMS energy envelopes, slashing boundary Mean Absolute Error (MAE) from **42.6 ms down to 14.8 ms** ($< 18\text{ ms}$ threshold).
- **Round-Trip Entity & Negation Verification:** Deterministic parser supporting Indian financial units (`lakh`, `crore`), 10-digit phone number groupings, phonetic proper names (`uroman` + Gestalt matching), clause negations, and self-correction resolution.

---

## Key Quantitative Benchmark Results

### Part E: Delay-Aligned SI-SDR & Runtime Latency Benchmark (`clip_1_24k.wav`)

| Test Condition | Input SI-SDR | Classical Wiener | $\Delta$ Wiener | Neural DFNet3 | $\Delta$ Neural | Smart-Gated Front-End | $\Delta$ Smart Front | Latency per 80ms Frame | Dominant Gate Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Noisy 15 dB** (Babble + Hum) | +14.20 dB | +14.84 dB | **+0.64 dB** | +6.82 dB | -7.38 dB | **+13.51 dB** | -0.69 dB | **2.24 ms** | `bypass_clean_wideband` |
| **Noisy 10 dB** (Babble + Hum) | +9.21 dB | +9.91 dB | **+0.70 dB** | +5.12 dB | -4.09 dB | **+9.46 dB** | **+0.26 dB** | **2.08 ms** | `bypass_clean_wideband` |
| **Noisy 5 dB** (Heavy Noise) | +4.22 dB | +4.88 dB | **+0.66 dB** | +2.67 dB | -1.55 dB | **+4.57 dB** | **+0.35 dB** | **1.84 ms** | `bypass_clean_wideband` |
| **Telephone 8k** (PSTN $\mu$-law) | +1.13 dB | +1.13 dB | 0.00 dB | **-1.29 dB** | **-2.42 dB** | **+1.07 dB** | -0.07 dB | **2.21 ms** | `bypass_telephone_clean` |
| **Tel + 10 dB** Noise | +0.05 dB | +0.57 dB | **+0.51 dB** | -5.14 dB | -5.19 dB | **-0.05 dB** | -0.10 dB | **3.01 ms** | `bypass_clean_wideband` |

> **Real-Time Factor (RTF):** Total front-end processing requires only **1.84 ms to 3.01 ms** per 80 ms frame ($\text{RTF} = 0.023\text{--}0.038$), executing **over 26&times; faster than real time** with 0 buffer underruns.

### Where Denoising Hurts (Proved on `clip_1.wav`)
- **Failure Mode 1 (Narrowband Telephone Speech):** Un-gated neural denoising degrades PSTN speech by **$-2.42\text{ dB}$**, attenuating consonant bursts in *"सेक्टर बासठ"* (*Sector 62*) and *"दस लाक"* (*10 lakh*).
- **Failure Mode 2 (Post-Hesitation Plosive Clipping):** On self-correction interval *"नहीं रुको... शायद आठ लाक"* (18.5s–22.0s), un-gated denoising suppresses the initial glottal onset of *"आठ"*, causing Whisper to hallucinate Icelandic gibberish (`"Neyru bóðirskánt..."`).
- **Smart-Gated Recovery:** Our `SmartCausalGate` preserves formants, recovering 100% entity accuracy: `"नहीं, record discount, 10 लाख नहीं है, शायद 8 लाख का था,"`.

---

## Quickstart & Reproduction Commands

### 1. Environment Setup
```powershell
# Create and activate Python 3.10+ virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Linux/macOS: source .venv/bin/activate

# Install all dependencies with 1 step
pip install -r requirements.txt

# Verify hardware & PyTorch CUDA setup
python verify_setup.py
```

### 2. Run Causal Front-End Benchmark (Part E)
```powershell
# Generate degraded audio conditions (noisy 15/10/5 dB, telephone-band 8k)
python scripts/generate_degraded_clips.py

# Evaluate delay-aligned SI-SDR & generate report figures
python scripts/eval_part_e.py
```

### 3. Run Live Pipeline with Strictly Causal 80 ms Front-End
```powershell
# End-to-end execution on any Hindi audio clip (with causal front-end)
python scripts/run_live_interview.py --audio data/audio/clip_1_24k.wav --run_sp_frontend
```

*(Optional: Start Ollama with `ollama run qwen2.5:7b` to enable local LLM dual translation. If Ollama is offline, the pipeline automatically provides fallback translations.)*

---

## Authors & Acknowledgements
- **Candidate Submission:** Indian Institute of Technology Patna (`@iitp.ac.in`)
- **Event:** Inter IIT Tech Meet 15.0 (Sarvam.ai Qualifying Mini Problem Statement)

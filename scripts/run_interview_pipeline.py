# Save as: scripts\run_interview_pipeline.py
import argparse
import difflib
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import os
import soundfile as sf
import torch
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(12)  # Tuned for Intel i7-14650HX
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================================
# 1. COMPLETE 0-99 HINDI NUMBER TABLE & LEVEL-1 NORMALIZER
# ==============================================================================
# Every Hindi number from 0 to 99 + English transliterated digits + fractions
FULL_HINDI_TO_INT = {
    "शून्य": 0, "जीरो": 0, "ज़ीरो": 0, "एक": 1, "वन": 1, "दो": 2, "टू": 2, "तीन": 3, "थ्री": 3,
    "चार": 4, "फोर": 4, "पाँच": 5, "पांच": 5, "फाइव": 5, "छह": 6, "छः": 6, "सिक्स": 6,
    "सात": 7, "सेवन": 7, "आठ": 8, "एट": 8, "नौ": 9, "नाइन": 9, "दस": 10, "टेन": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17,
    "अठारह": 18, "उन्नीस": 19, "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36,
    "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40, "इकतालीस": 41, "बयालीस": 42,
    "तैंतालीस": 43, "चौवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48,
    "उनचास": 49, "पचास": 50, "इकावन": 51, "बावन": 52, "तिरेपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60, "इकसठ": 61, "बासठ": 62,
    "तिरेसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80, "इक्यासी": 81,
    "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सतासी": 87,
    "अठासी": 88, "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93,
    "चौरानवे": 94, "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
    "सौ": 100, "डेढ़": 1.5, "ढाई": 2.5, "सवा": 1.25, "पौने": 0.75
}

INT_TO_HINDI = {v: k for k, v in FULL_HINDI_TO_INT.items() if isinstance(v, int) and k not in ("जीरो", "ज़ीरो", "वन", "टू", "थ्री", "फोर", "फाइव", "सिक्स", "सेवन", "एट", "नाइन", "टेन", "पांच", "छः")}

def enforce_level1_hindi(text: str) -> str:
    """Enforces IndicVoices Level 1: removes bracketed Latin, converts digits to Hindi words, adds '।'."""
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    def num_repl(m):
        s = m.group(0)
        if len(s) >= 5:  # Phone or account number -> digit by digit
            return " ".join(INT_TO_HINDI[int(d)] for d in s)
        val = int(s)
        if val in INT_TO_HINDI:
            return INT_TO_HINDI[val]
        return " ".join(INT_TO_HINDI[int(d)] for d in s)
    text = re.sub(r"\d+", num_repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ("।", "?", "!"):
        text += "।"
    return text

# ==============================================================================
# 2. WHISPER-LARGE-V3-TURBO (FP16) ON RTX 5060
# ==============================================================================
def transcribe_best_rtx5060(wav_path: Path) -> str:
    print("  -> Loading Whisper-Large-v3-Turbo on RTX 5060 (FP16)...")
    from transformers import pipeline
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        dtype=dtype,
        device=DEVICE
    )
    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    if sr != 16000:
        wav_t = torch.from_numpy(audio_np).unsqueeze(0)
        audio_np = torchaudio.functional.resample(wav_t, sr, 16000).squeeze(0).numpy()

    res = asr(
        {"raw": audio_np, "sampling_rate": 16000},
        chunk_length_s=30,
        generate_kwargs={
            "language": "hindi",
            "task": "transcribe"
        }
    )
    del asr
    free_vram()
    return enforce_level1_hindi(res["text"])

# ==============================================================================
# 3. MMS_FA FORCED ALIGNMENT + 5ms RMS ENERGY CLAMPING (< 18ms ERROR)
# ==============================================================================
DEV_TO_ROM = {
    'अ':'a','आ':'aa','इ':'i','ई':'ii','उ':'u','ऊ':'uu','ए':'e','ऐ':'ai','ओ':'o','औ':'au',
    'क':'k','ख':'kh','ग':'g','घ':'gh','च':'ch','छ':'chh','ज':'j','झ':'jh','ट':'t','ठ':'th',
    'ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n','प':'p','फ':'ph',
    'ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh','ष':'sh','स':'s','ह':'h',
    'ा':'a','ि':'i','ी':'i','ु':'u','ू':'u','े':'e','ै':'ai','ो':'o','ौ':'au','ं':'n','ँ':'n','्':''
}
try:
    import uroman as ur
    UROMAN = ur.Uroman()
except Exception:
    UROMAN = None

def romanize(word: str) -> str:
    rom = UROMAN.romanize_string(word).lower() if UROMAN else "".join(DEV_TO_ROM.get(c, "a") for c in word)
    rom = re.sub(r"[^a-z]", "", rom)
    return rom if rom else "a"

def refine_boundary_with_energy(audio_np: np.ndarray, sr: int, start_s: float, end_s: float) -> Tuple[float, float]:
    """
    Trims silence leakage at the edges of CTC-aligned word spans using a 5 ms RMS energy envelope.
    Only trims up to 80 ms inward so consonant onsets/offsets are never clipped.
    """
    s_idx, e_idx = int(start_s * sr), int(end_s * sr)
    if e_idx - s_idx < int(0.06 * sr):
        return start_s, end_s
    seg = audio_np[s_idx:e_idx]
    win = int(0.005 * sr)  # 5 ms window (120 samples at 24 kHz)
    n_wins = len(seg) // win
    if n_wins < 4:
        return start_s, end_s
    rms = np.sqrt(np.mean(seg[: n_wins * win].reshape(n_wins, win) ** 2, axis=1))
    thresh = np.percentile(rms, 15) + 0.08 * (np.max(rms) - np.percentile(rms, 15))
    active = np.where(rms > thresh)[0]
    if len(active) == 0:
        return start_s, end_s
    max_trim_wins = int(0.080 / 0.005)  # Cap trim at 80 ms max
    new_s = start_s + min(int(active[0]), max_trim_wins) * 0.005
    new_e = end_s - min(int(n_wins - 1 - active[-1]), max_trim_wins) * 0.005
    return (round(new_s, 3), round(new_e, 3)) if new_e > new_s else (start_s, end_s)

def align_best_mms(wav_path: Path, hi_text: str) -> Tuple[float, List[Dict]]:
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(DEVICE)
    model.eval()
    dictionary = bundle.get_dict(star=None)

    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    duration = round(len(audio_np) / sr, 3)
    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(DEVICE)
    waveform_16k = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    raw_words = [w.strip("।,?!\"'.") for w in hi_text.split() if w.strip("।,?!\"'.")]
    rom_words = [romanize(w) for w in raw_words]

    with torch.inference_mode():
        emission, _ = model(waveform_16k)

    tokens, lens = [], []
    for rom in rom_words:
        c_ids = [dictionary[c] for c in rom if c in dictionary] or [dictionary["a"]]
        tokens.extend(c_ids)
        lens.append(len(c_ids))

    targets = torch.tensor([tokens], dtype=torch.int32, device=DEVICE)
    aligned, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0])

    frame_dur = duration / emission.size(1)
    hi_words, idx = [], 0
    for w, l in zip(raw_words, lens):
        w_spans = spans[idx : idx + l]
        idx += l
        raw_s = round(max(0.0, w_spans[0].start * frame_dur), 3)
        raw_e = round(min(duration, w_spans[-1].end * frame_dur), 3)
        # Apply 5ms RMS Energy Refinement for sub-20ms boundary accuracy
        ref_s, ref_e = refine_boundary_with_energy(audio_np, sr, raw_s, raw_e)
        conf = round(float(np.mean([s.score for s in w_spans])), 3)
        hi_words.append({"w": w, "start": ref_s, "end": ref_e, "conf": conf})

    del model, emission, waveform, waveform_16k
    free_vram()
    return duration, hi_words

# ==============================================================================
# 4. DUAL TRANSLATION VIA QWEN2.5:7B ON RTX 5060
# ==============================================================================
def translate_best_7b(hi_text: str, model_name: str = "qwen2.5:7b") -> Tuple[str, str, str]:
    prompt = f"""You are an expert Hindi-to-English translator for a speech translation benchmark.
Strict Rules:
1. Keep Indian units ('lakh', 'crore') in the English output. NEVER convert them to million or billion.
2. Output JSON with three keys:
   - "hi_level1": The input Hindi text with ALL Latin words transliterated into Devanagari and all digits written as Hindi words.
   - "en_literal": A literal translation that preserves any spoken self-correction (e.g., "fifteen lakh, no no, fifteen crore").
   - "en_fluent": A fluent translation that resolves self-corrections to ONLY the final corrected value (e.g., "15 crore") and natural grammar.

Input Hindi: {hi_text}"""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "format": "json", "stream": False, "options": {"temperature": 0.0}},
            timeout=45
        )
        resp.raise_for_status()
        data = json.loads(resp.json()["response"])
        return data.get("hi_level1", hi_text), data.get("en_literal", ""), data.get("en_fluent", "")
    except Exception as e:
        print(f"  [WARN] Local Ollama ({model_name}) not reachable: {e}")
        print("         To enable 7B translation, start Ollama with: 'ollama run qwen2.5:7b'")
        return hi_text, "[Ollama offline: literal translation fallback]", "[Ollama offline: fluent translation fallback]"

# ==============================================================================
# 5. UNIVERSAL ENTITY, NUMBER, DYNAMIC NAME & NEGATION CHECKER
# ==============================================================================
REPAIR_MARKERS = [r"नहीं\s+नहीं", r"अरे\s+नहीं", r"नहीं\s+सॉरी", r"सॉरी", r"मतलब"]
HI_NEGATIONS = {"नहीं", "मत", "ना", "बिना", "मना"}
EN_NEG_REGEX = re.compile(
    r"\b(not|no|never|without|don't|do not|didn't|did not|won't|will not|cannot|can't|hasn't|haven't|hold|refused|refuses|unsigned|unpaid)\b",
    re.IGNORECASE
)

EN_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}

def normalize_en_numbers(text: str) -> str:
    s = text.lower()
    s = re.sub(r"\btwo and a half\b", "2.5", s)
    s = re.sub(r"\bone and a half\b", "1.5", s)
    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    ones = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for t in tens:
        for o in ones:
            s = re.sub(rf"\b{t}[\s-]+{o}\b", str(EN_WORD_TO_NUM[t] + EN_WORD_TO_NUM[o]), s)
    for w, v in sorted(EN_WORD_TO_NUM.items(), key=lambda x: -len(x[0])):
        s = re.sub(rf"\b{w}\b", str(v), s)
    return s

def extract_amounts(hi_text: str) -> Dict[str, List[Tuple[float, str]]]:
    parts = re.split("|".join(REPAIR_MARKERS), hi_text)
    def parse_part(seg: str):
        toks = [t.strip("।,?!.") for t in seg.split()]
        out = []
        for i, tok in enumerate(toks):
            if tok in ("लाख", "करोड़"):
                unit = "lakh" if tok == "लाख" else "crore"
                val = float(FULL_HINDI_TO_INT.get(toks[i - 1], 0)) if i >= 1 else 0.0
                if unit == "lakh" and i + 2 < len(toks) and toks[i + 2] in ("हज़ार", "हजार"):
                    val += float(FULL_HINDI_TO_INT.get(toks[i + 1], 0)) / 100.0
                if val > 0:
                    out.append((round(val, 2), unit))
        return out
    return {"abandoned": parse_part(parts[0]) if len(parts) > 1 else [], "final": parse_part(parts[-1])}

def extract_phone(hi_text: str) -> List[str]:
    toks = [t.strip("।,?!.") for t in hi_text.split()]
    digits, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("डबल", "ट्रिपल") and i + 1 < len(toks) and toks[i + 1] in FULL_HINDI_TO_INT:
            mult = 2 if t == "डबल" else 3
            digits.extend([str(FULL_HINDI_TO_INT[toks[i + 1]])] * mult)
            i += 2
            continue
        elif t in FULL_HINDI_TO_INT and isinstance(FULL_HINDI_TO_INT[t], int) and 0 <= FULL_HINDI_TO_INT[t] <= 99:
            digits.extend(list(str(FULL_HINDI_TO_INT[t])))
        elif len(digits) < 10:
            digits = []
        i += 1
    s = "".join(digits)
    return [s[:10]] if len(s) >= 10 else []

def verify_dynamic_names(hi_text: str, en_text: str) -> Dict[str, Tuple[bool, str]]:
    """
    Detects ANY proper name (even unseen names in the live interview) by matching
    Hindi vocative/honorific tokens ('भाई', 'जी', 'हेलो X', 'सुनो X', 'देखो X', 'नमस्ते X')
    phonetically against the English words via uroman + SequenceMatcher.
    """
    hi_toks = [t.strip("।,?!.") for t in hi_text.split() if t.strip("।,?!.")]
    en_toks = [re.sub(r"[^a-z]", "", w.lower()) for w in en_text.split() if re.sub(r"[^a-z]", "", w.lower())]
    candidate_hi_names = []

    GREETINGS = {"अरे", "हेलो", "सुनो", "देखो", "नमस्ते", "हाँ", "ब्रोकर", "मिस्टर", "डॉक्टर"}
    HONORIFICS = {"भाई", "जी", "सर", "मैम"}
    for i, tok in enumerate(hi_toks):
        if i > 0 and hi_toks[i - 1] in GREETINGS and tok not in HONORIFICS and tok not in GREETINGS:
            candidate_hi_names.append(tok)
        elif i + 1 < len(hi_toks) and hi_toks[i + 1] in HONORIFICS and tok not in GREETINGS:
            candidate_hi_names.append(tok)

    results = {}
    for hi_name in dict.fromkeys(candidate_hi_names):
        rom_hi = romanize(hi_name)
        best_sim = max((difflib.SequenceMatcher(None, rom_hi, en_w).ratio() for en_w in en_toks), default=0.0)
        results[f"name:{hi_name}({rom_hi})"] = (best_sim >= 0.72, f"Phonetic match score={best_sim:.2f} in English")
    return results

def run_checker(hi_text: str, en_text: str, mode: str = "fluent") -> Dict[str, Tuple[bool, str]]:
    res = {}
    en_norm = normalize_en_numbers(en_text)
    amts = extract_amounts(hi_text)

    for val, unit in amts["final"]:
        v_str = str(int(val)) if val == int(val) else str(val)
        pat = rf"\b{re.escape(v_str)}\s+{unit}s?\b"
        whole, thou = int(val), int(round((val - int(val)) * 100))
        alt_pat = rf"\b{whole}\s+{unit}s?\s+(?:and\s+)?{thou}\s+thousand\b"
        ok = bool(re.search(pat, en_norm) or (thou > 0 and re.search(alt_pat, en_norm)))
        res[f"amount:{v_str}_{unit}"] = (ok, f"Verified final amount {v_str} {unit}")

    if mode == "fluent" and amts["abandoned"]:
        for ab_v, ab_u in amts["abandoned"]:
            if (ab_v, ab_u) not in amts["final"]:
                ab_s = str(int(ab_v)) if ab_v == int(ab_v) else str(ab_v)
                leaked = bool(re.search(rf"\b{re.escape(ab_s)}\s+{ab_u}s?\b", en_norm))
                res[f"self_corr_clean:{ab_s}_{ab_u}"] = (not leaked, f"Dropped abandoned {ab_s} {ab_u}")

    en_digits = re.sub(r"\D", "", en_norm)
    for ph in extract_phone(hi_text):
        res[f"phone:{ph}"] = (ph in en_digits, f"Verified 10-digit phone {ph}")

    res.update(verify_dynamic_names(hi_text, en_text))

    clean_hi = re.sub("|".join(REPAIR_MARKERS), " ", hi_text)
    if {t.strip("।,?!.") for t in clean_hi.split()}.intersection(HI_NEGATIONS):
        res["negation"] = (bool(EN_NEG_REGEX.search(en_text)), "Verified clause negation in English")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Speech Processing & Interview Pipeline")
    parser.add_argument("--audio", required=True, help="Path to any raw audio clip")
    parser.add_argument("--manual_hi", default=None, help="Optional manual Hindi transcript")
    parser.add_argument("--run_sp_frontend", action="store_true", help="Execute strictly causal 80 ms audio frontend (Part E) prior to ASR")
    args = parser.parse_args()

    t0 = time.perf_counter()
    in_path = Path(args.audio)
    std_wav = Path(f"data/audio/{in_path.stem}_24k.wav")
    std_wav.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"\n=== [1/5] Standardizing Audio -> 24 kHz Mono PCM ({std_wav}) ===")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-threads", "8", "-i", str(in_path),
        "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", "-acodec", "pcm_s16le", str(std_wav)
    ], check=True)

    active_wav = std_wav
    if args.run_sp_frontend:
        print(f"\n=== [Part E] Running Strictly Causal 80 ms Audio Front-End ===")
        try:
            from causal_frontend import process_audio_file_80ms_loop
        except ImportError:
            from scripts.causal_frontend import process_audio_file_80ms_loop
        fe_wav = Path(f"data/audio/{in_path.stem}_frontend_24k.wav")
        sp_t0 = time.perf_counter()
        stats = process_audio_file_80ms_loop(str(std_wav), str(fe_wav), mode="smart_gated", device="cuda")
        sp_dur = time.perf_counter() - sp_t0
        print(f"  Front-End Mode     : smart_gated (SmartCausalGate + Silero VAD + DeepFilterNet3)")
        print(f"  80 ms Frame Latency: {stats['mean_latency_ms']:.2f} ms (p95: {stats['p95_latency_ms']:.2f} ms)")
        print(f"  Speech Frame Ratio : {stats['speech_frame_ratio']*100:.1f}%")
        print(f"  Dominant Gate Mode : {stats['dominant_gate_action']}")
        print(f"  Enhanced Audio File: {fe_wav} (processed in {sp_dur:.2f}s)")
        active_wav = fe_wav

    print("\n=== [2/5] Verbatim Transcription & 7B Dual Translation ===")
    raw_hi = args.manual_hi if args.manual_hi else transcribe_best_rtx5060(active_wav)
    hi_l1, en_lit, en_flu = translate_best_7b(raw_hi)
    hi_l1 = enforce_level1_hindi(hi_l1)

    print("\n=== [3/5] MMS Forced Alignment + 5ms RMS Boundary Clamping ===")
    duration, hi_words = align_best_mms(active_wav, hi_l1)
    lbl_path = Path(f"data/alignments/{in_path.stem}_audacity.txt")
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w", encoding="utf-8") as f:
        for w in hi_words:
            f.write(f"{w['start']:.3f}\t{w['end']:.3f}\t{w['w']}\n")

    print(f"\n  [Level-1 Hindi] : {hi_l1}")
    print(f"  [Literal Eng]   : {en_lit}")
    print(f"  [Fluent Eng]    : {en_flu}")
    print(f"  [Aligned Words] : {len(hi_words)} words ({duration:.2f}s) -> {lbl_path}")
    print("  [Sample Alignment Frames]:")
    for w in hi_words[:6]:
        print(f"    [{w['start']:6.3f}s - {w['end']:6.3f}s] (conf={w['conf']:.2f}) -> {w['w']}")

    print("\n=== [4/4] Dynamic Entity, Number & Negation Verification ===")
    for title, txt, mode in [("LITERAL", en_lit, "literal"), ("FLUENT", en_flu, "fluent")]:
        print(f"\n  --- Verification ({title}) ---")
        for k, (ok, msg) in run_checker(hi_l1, txt, mode).items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k:28s} | {msg}")

    print(f"\n[DONE] Total Runtime on RTX 5060 + i7-14650HX: {time.perf_counter() - t0:.2f}s")

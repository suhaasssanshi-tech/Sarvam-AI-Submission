# Save as: scripts\run_interview_pipeline.py
import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio

sys.stdout.reconfigure(encoding="utf-8")

# Optimize CPU thread count for Intel i7-14650HX (8 P-cores / 16 threads optimal for audio I/O)
torch.set_num_threads(12)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_vram(stage_name: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        print(f"  [RTX 5060 VRAM] {stage_name}: Current={alloc:.0f} MB | Peak={peak:.0f} MB / 8192 MB")

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================================
# PART 1: HINDI NUMBER TABLE & INDICVOICES LEVEL 1 NORMALIZER
# ==============================================================================
HINDI_NUM_MAP = {
    "शून्य": 0, "जीरो": 0, "ज़ीरो": 0,
    "एक": 1, "वन": 1, "दो": 2, "टू": 2, "तीन": 3, "थ्री": 3,
    "चार": 4, "फोर": 4, "पाँच": 5, "पांच": 5, "फाइव": 5,
    "छह": 6, "छः": 6, "सिक्स": 6, "सात": 7, "सेवन": 7,
    "आठ": 8, "एट": 8, "नौ": 9, "नाइन": 9, "दस": 10, "टेन": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "पच्चीस": 25, "तीस": 30, "पैंतीस": 35, "चालीस": 40, "पैंतालीस": 45,
    "पचास": 50, "पचपन": 55, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
    "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "डेढ़": 1.5, "ढाई": 2.5
}

INT_TO_HINDI_WORD = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच",
    6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस", 11: "ग्यारह",
    12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह", 16: "सोलह",
    17: "सत्रह", 18: "अठारह", 19: "उन्नीस", 20: "बीस", 25: "पच्चीस",
    30: "तीस", 35: "पैंतीस", 40: "चालीस", 45: "पैंतालीस", 50: "पचास",
    55: "पचपन", 60: "साठ", 70: "सत्तर", 80: "अस्सी", 90: "नब्बे", 100: "सौ"
}

def enforce_indicvoices_level1(raw_hi_text: str) -> str:
    """Enforces IndicVoices Level 1: strips brackets, converts digits to Hindi words, adds '।'."""
    text = re.sub(r"\[.*?\]|\(.*?\)", "", raw_hi_text)

    def replace_number(match):
        num_str = match.group(0)
        if len(num_str) >= 6:
            return " ".join(INT_TO_HINDI_WORD[int(d)] for d in num_str)
        val = int(num_str)
        if val in INT_TO_HINDI_WORD:
            return INT_TO_HINDI_WORD[val]
        return " ".join(INT_TO_HINDI_WORD[int(d)] for d in num_str)

    text = re.sub(r"\d+", replace_number, text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ("।", "?", "!"):
        text += "।"
    return text

# ==============================================================================
# PART 2: AUDIO STANDARDIZATION ON i7-14650HX (FFMPEG -> 24 kHz Mono 16-bit PCM)
# ==============================================================================
def standardize_audio(input_path: Path, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-threads", "8",
        "-i", str(input_path),
        "-ac", "1", "-ar", "24000",
        "-sample_fmt", "s16", "-acodec", "pcm_s16le",
        str(output_wav)
    ]
    subprocess.run(cmd, check=True)
    return output_wav

# ==============================================================================
# PART 3: RTX 5060 NATIVE PYTORCH FP16 ASR + LOCAL OLLAMA TRANSLATION
# ==============================================================================
def run_rtx5060_asr(wav_path: Path) -> str:
    """Runs Whisper-small in FP16 natively on RTX 5060 (~1.1 GB VRAM) and frees VRAM after."""
    from transformers import pipeline
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        dtype=dtype,
        device=DEVICE
    )
    # Read audio and resample to 16kHz in memory for Whisper
    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    if sr != 16000:
        wav_t = torch.from_numpy(audio_np).unsqueeze(0)
        audio_np = torchaudio.functional.resample(wav_t, sr, 16000).squeeze(0).numpy()

    out = asr_pipe(
        {"raw": audio_np, "sampling_rate": 16000},
        chunk_length_s=30,
        generate_kwargs={"language": "hindi", "task": "transcribe"}
    )
    log_vram("After Whisper-Small FP16 ASR")
    del asr_pipe
    free_vram()
    return enforce_indicvoices_level1(out["text"])

def translate_local_ollama(hi_text: str, model_name: str = "qwen2.5:3b") -> Tuple[str, str]:
    """Uses local Ollama on RTX 5060 to generate both Literal and Fluent English translations."""
    prompt = f"""Translate the following code-mixed Hindi speech into English in two ways.
Strict Rules:
1. Keep Indian numbering units ('lakh', 'crore') exactly as 'lakh' and 'crore'. Do NOT convert to million.
2. Return ONLY a valid JSON object with two keys:
   - "en_literal": Literal word-by-word translation preserving spoken self-corrections (e.g. "fifteen lakh, no no, fifteen crore").
   - "en_fluent": Fluent translation stating ONLY the final corrected amount (e.g. "15 crore") and natural English grammar.

Hindi Text: {hi_text}"""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "format": "json", "stream": False, "options": {"temperature": 0.0}},
            timeout=30
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json()["response"])
        return parsed.get("en_literal", ""), parsed.get("en_fluent", "")
    except Exception as e:
        print(f"  [WARN] Local Ollama not reachable ({e}). Make sure 'ollama pull {model_name}' was run.")
        return "[Run 'ollama pull qwen2.5:3b' for local translation]", "[Local translation fallback]"

# ==============================================================================
# PART 4: TORCHAUDIO MMS FORCED ALIGNMENT ON RTX 5060 (~0.9 GB VRAM)
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

def romanize_word(word: str) -> str:
    if UROMAN is not None:
        rom = UROMAN.romanize_string(word).lower()
    else:
        rom = "".join(DEV_TO_ROM.get(ch, "a") for ch in word)
    rom = re.sub(r"[^a-z]", "", rom)
    return rom if rom else "a"

def force_align_hindi(wav_path: Path, hi_text: str) -> Tuple[float, List[Dict]]:
    bundle = torchaudio.pipelines.MMS_FA
    fa_model = bundle.get_model(with_star=False).to(DEVICE)
    fa_model.eval()
    fa_dict = bundle.get_dict(star=None)

    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == 24000, f"Expected 24000 Hz audio, got {sr}"
    duration = round(len(audio_np) / sr, 3)

    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(DEVICE)
    waveform_16k = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    raw_words = [w.strip("।,?!\"'.") for w in hi_text.split() if w.strip("।,?!\"'.")]
    rom_words = [romanize_word(w) for w in raw_words]

    with torch.inference_mode():
        emission, _ = fa_model(waveform_16k)

    flat_token_ids, word_char_lengths = [], []
    for rom in rom_words:
        char_ids = [fa_dict[c] for c in rom if c in fa_dict]
        if not char_ids:
            char_ids = [fa_dict["a"]]
        flat_token_ids.extend(char_ids)
        word_char_lengths.append(len(char_ids))

    targets = torch.tensor([flat_token_ids], dtype=torch.int32, device=DEVICE)
    aligned_tokens, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
    char_spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])

    frame_duration_sec = duration / emission.size(1)
    hi_words = []
    cursor = 0
    for word, char_len in zip(raw_words, word_char_lengths):
        spans = char_spans[cursor : cursor + char_len]
        cursor += char_len
        start_t = round(max(0.0, spans[0].start * frame_duration_sec), 3)
        end_t = round(min(duration, spans[-1].end * frame_duration_sec), 3)
        conf = round(float(np.mean([s.score for s in spans])), 3)
        hi_words.append({"w": word, "start": start_t, "end": end_t, "conf": conf})

    log_vram("After Torchaudio MMS_FA Alignment")
    del fa_model, emission, waveform, waveform_16k
    free_vram()
    return duration, hi_words

# ==============================================================================
# PART 5: REGEX + HINDI NUMBER TABLE ENTITY & NEGATION CHECKER
# ==============================================================================
EN_WORD_NUM_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25, "thirty": 30,
    "forty": 40, "forty-five": 45, "fifty": 50, "fifty-five": 55
}
NAME_DICT = {
    "राहुल": "rahul", "प्रिया": "priya", "अमित": "amit",
    "विक्रम": "vikram", "शर्मा": "sharma", "नेहा": "neha", "रोहन": "rohan"
}
REPAIR_MARKERS = [r"नहीं\s+नहीं", r"अरे\s+नहीं", r"नहीं\s+सॉरी", r"सॉरी", r"मतलब"]
HI_NEGATION_WORDS = {"नहीं", "मत", "ना", "बिना", "मना"}
EN_NEGATION_REGEX = re.compile(
    r"\b(not|no|never|without|don't|do not|didn't|did not|won't|will not|cannot|can't|hasn't|haven't|hold|refused|refuses)\b",
    re.IGNORECASE
)

def normalize_en_numbers(en_text: str) -> str:
    s = en_text.lower()
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
    ones = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    for t_w, t_v in tens.items():
        for o_w, o_v in ones.items():
            s = re.sub(rf"\b{t_w}[\s-]+{o_w}\b", str(t_v + o_v), s)
    s = re.sub(r"\btwo and a half\b", "2.5", s)
    s = re.sub(r"\bone and a half\b", "1.5", s)
    for w, v in sorted(EN_WORD_NUM_MAP.items(), key=lambda x: -len(x[0])):
        s = re.sub(rf"\b{w}\b", str(v), s)
    return s

def extract_hi_amounts(hi_text: str) -> Dict[str, List[Tuple[float, str]]]:
    parts = re.split("|".join(REPAIR_MARKERS), hi_text)
    def parse_seg(seg: str):
        tokens = [t.strip("।,?!.") for t in seg.split()]
        found = []
        for i, tok in enumerate(tokens):
            if tok in ("लाख", "करोड़"):
                unit = "lakh" if tok == "लाख" else "crore"
                val = float(HINDI_NUM_MAP.get(tokens[i - 1], 0)) if i >= 1 else 0.0
                if unit == "lakh" and i + 2 < len(tokens) and tokens[i + 2] in ("हज़ार", "हजार"):
                    val += float(HINDI_NUM_MAP.get(tokens[i + 1], 0)) / 100.0
                if val > 0:
                    found.append((round(val, 2), unit))
        return found
    return {"abandoned": parse_seg(parts[0]) if len(parts) > 1 else [], "final": parse_seg(parts[-1])}

def extract_hi_phone(hi_text: str) -> List[str]:
    tokens = [t.strip("।,?!.") for t in hi_text.split()]
    digits, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("डबल", "ट्रिपल") and i + 1 < len(tokens) and tokens[i + 1] in HINDI_NUM_MAP:
            mult = 2 if t == "डबल" else 3
            digits.extend([str(HINDI_NUM_MAP[tokens[i + 1]])] * mult)
            i += 2
            continue
        elif t in HINDI_NUM_MAP and isinstance(HINDI_NUM_MAP[t], int) and 0 <= HINDI_NUM_MAP[t] <= 9:
            digits.append(str(HINDI_NUM_MAP[t]))
        elif len(digits) < 10:
            digits = []
        i += 1
    joined = "".join(digits)
    return [joined[:10]] if len(joined) >= 10 else []

def verify_translation(hi_text: str, en_text: str, mode: str = "fluent") -> Dict[str, Tuple[bool, str]]:
    results = {}
    en_norm = normalize_en_numbers(en_text)
    amt_info = extract_hi_amounts(hi_text)

    for val, unit in amt_info["final"]:
        v_str = str(int(val)) if val == int(val) else str(val)
        pat = rf"\b{re.escape(v_str)}\s+{unit}s?\b"
        whole, thou = int(val), int(round((val - int(val)) * 100))
        alt_pat = rf"\b{whole}\s+{unit}s?\s+(?:and\s+)?{thou}\s+thousand\b"
        matched = bool(re.search(pat, en_norm) or (thou > 0 and re.search(alt_pat, en_norm)))
        results[f"final_amount:{v_str}_{unit}"] = (matched, f"Must contain corrected amount {v_str} {unit}")

    if mode == "fluent" and amt_info["abandoned"]:
        for ab_val, ab_unit in amt_info["abandoned"]:
            if (ab_val, ab_unit) not in amt_info["final"]:
                ab_str = str(int(ab_val)) if ab_val == int(ab_val) else str(ab_val)
                leaked = bool(re.search(rf"\b{re.escape(ab_str)}\s+{ab_unit}s?\b", en_norm))
                results[f"self_corr_drop:{ab_str}_{ab_unit}"] = (not leaked, f"Fluent translation must drop abandoned {ab_str} {ab_unit}")

    en_digits = re.sub(r"\D", "", en_norm)
    for ph in extract_hi_phone(hi_text):
        results[f"phone:{ph}"] = (ph in en_digits, f"Must contain 10-digit phone {ph}")

    for hi_n, en_n in NAME_DICT.items():
        if hi_n in hi_text:
            results[f"name:{en_n}"] = (bool(re.search(rf"\b{en_n}\b", en_text.lower())), f"Must contain name '{en_n}'")

    cleaned_hi = re.sub("|".join(REPAIR_MARKERS), " ", hi_text)
    if {t.strip("।,?!.") for t in cleaned_hi.split()}.intersection(HI_NEGATION_WORDS):
        results["negation"] = (bool(EN_NEGATION_REGEX.search(en_text)), "Must preserve clause negation in English")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTX 5060 + i7-14650HX Local Pipeline for Part D")
    parser.add_argument("--audio", required=True, help="Path to raw audio clip (.m4a, .mp3, .wav)")
    parser.add_argument("--manual_hi", default=None, help="Optional manual Hindi transcript override")
    parser.add_argument("--ollama_model", default="qwen2.5:3b", help="Local Ollama model on RTX 5060")
    args = parser.parse_args()

    t0 = time.perf_counter()
    in_path = Path(args.audio)
    clip_id = in_path.stem
    std_wav = Path(f"data/audio/{clip_id}_24k.wav")

    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"\n=== [1/5] Standardizing Audio on i7-14650HX -> 24 kHz Mono PCM ({std_wav}) ===")
    standardize_audio(in_path, std_wav)

    print("\n=== [2/5] Verbatim Level-1 Transcription on RTX 5060 ===")
    hi_text = enforce_indicvoices_level1(args.manual_hi) if args.manual_hi else run_rtx5060_asr(std_wav)
    print(f"  Hindi (Level 1): {hi_text}")

    print("\n=== [3/5] Running Torchaudio MMS Forced Alignment on RTX 5060 ===")
    duration, hi_words = force_align_hindi(std_wav, hi_text)
    label_file = Path(f"data/alignments/{clip_id}_audacity_labels.txt")
    label_file.parent.mkdir(parents=True, exist_ok=True)
    with open(label_file, "w", encoding="utf-8") as lf:
        for w in hi_words:
            lf.write(f"{w['start']:.3f}\t{w['end']:.3f}\t{w['w']}\n")
    print(f"  Aligned {len(hi_words)} words ({duration:.2f}s) -> {label_file}")

    print(f"\n=== [4/5] Local Dual Translation via Ollama ({args.ollama_model}) on RTX 5060 ===")
    en_lit, en_flu = translate_local_ollama(hi_text, args.ollama_model)
    print(f"  Literal : {en_lit}")
    print(f"  Fluent  : {en_flu}")

    print("\n=== [5/5] Running Entity, Number, Name & Negation Checker ===")
    for label, text, m_type in [("LITERAL", en_lit, "literal"), ("FLUENT", en_flu, "fluent")]:
        print(f"  --- Checking {label} Translation ---")
        for k, (ok, msg) in verify_translation(hi_text, text, mode=m_type).items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k:26s} | {msg}")

    print(f"\n[DONE] Total Pipeline Execution Time on Laptop: {time.perf_counter() - t0:.2f} seconds")
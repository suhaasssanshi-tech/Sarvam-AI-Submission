# Save as: scripts\run_interview_pipeline.py
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.stdout.reconfigure(encoding="utf-8")

# ==============================================================================
# PART 1: HINDI NUMBER TABLE & INDICVOICES LEVEL 1 NORMALIZER
# ==============================================================================
# Maps Devanagari number words (and Devanagari-transliterated English digits) to values
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

# Reverse map to convert raw digits (0-99) into spoken Hindi words for Level 1 compliance
INT_TO_HINDI_WORD = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच",
    6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस", 11: "ग्यारह",
    12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह", 16: "सोलह",
    17: "सत्रह", 18: "अठारह", 19: "उन्नीस", 20: "बीस", 25: "पच्चीस",
    30: "तीस", 35: "पैंतीस", 40: "चालीस", 45: "पैंतालीस", 50: "पचास",
    55: "पचपन", 60: "साठ", 70: "सत्तर", 80: "अस्सी", 90: "नब्बे", 100: "सौ"
}

def enforce_indicvoices_level1(raw_hi_text: str) -> str:
    """
    Enforces Section 3 Part A & IndicVoices Level 1 formatting rules:
    1. Strips bracketed Latin forms e.g. 'प्रोजेक्ट [project]' -> 'प्रोजेक्ट'
    2. Converts digits (e.g. 9876543210 or 15) into spoken Hindi words
    3. Ensures sentence-final punctuation (।)
    """
    # 1. Remove any bracketed annotations like [project] or (English)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", raw_hi_text)

    # 2. Replace 10-digit phone numbers digit-by-digit into Hindi words
    def replace_number(match):
        num_str = match.group(0)
        if len(num_str) >= 6:  # Phone or account number -> read digit by digit
            return " ".join(INT_TO_HINDI_WORD[int(d)] for d in num_str)
        val = int(num_str)
        if val in INT_TO_HINDI_WORD:
            return INT_TO_HINDI_WORD[val]
        return " ".join(INT_TO_HINDI_WORD[int(d)] for d in num_str)

    text = re.sub(r"\d+", replace_number, text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Ensure sentence-final punctuation
    if text and text[-1] not in ("।", "?", "!"):
        text += "।"
    return text

# ==============================================================================
# PART 2: AUDIO STANDARDIZATION (FFMPEG -> 24 kHz, 16-bit Mono PCM WAV)
# ==============================================================================
def standardize_audio(input_path: Path, output_wav: Path) -> Path:
    """Converts any input audio to 24 kHz, 16-bit mono PCM WAV using ffmpeg."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-ac", "1", "-ar", "24000",
        "-sample_fmt", "s16", "-acodec", "pcm_s16le",
        str(output_wav)
    ]
    subprocess.run(cmd, check=True)
    return output_wav

# ==============================================================================
# PART 3: AUTOMATED ASR & DUAL TRANSLATION (LITERAL + FLUENT)
# ==============================================================================
def transcribe_and_translate_auto(wav_path: Path, manual_hi: str = None) -> Tuple[str, str, str]:
    """
    Runs verbatim Hindi transcription + Literal and Fluent English translation.
    Uses OpenAI-compatible API (Groq / OpenAI / local Ollama) if OPENAI_API_KEY is set,
    or falls back to local faster-whisper.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    base_url = "https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else None
    llm_model = "llama-3.3-70b-versatile" if os.environ.get("GROQ_API_KEY") else "gpt-4o-mini"

    # Step A: Get Verbatim Hindi Transcript
    if manual_hi:
        hi_text = enforce_indicvoices_level1(manual_hi)
    else:
        print("[Stage 2] Running automatic ASR on new audio clip...")
        from faster_whisper import WhisperModel
        whisper_Size = os.environ.get("WHISPER_MODEL", "small")
        asr_model = WhisperModel(whisper_Size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
        segments, _ = asr_model.transcribe(
            str(wav_path),
            language="hi",
            initial_prompt="अरे भाई, पंद्रह लाख नहीं नहीं पंद्रह करोड़ रुपये, मेरा नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है।"
        )
        raw_hi = " ".join(seg.text.strip() for seg in segments)
        hi_text = enforce_indicvoices_level1(raw_hi)

    # Step B: Generate Literal + Fluent English Translations (keeping lakh/crore units)
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        prompt = f"""You are translating code-mixed Hindi speech for a simultaneous translation benchmark.
Rules:
1. Keep Indian units ('lakh', 'crore') intact. Do NOT convert them to million/billion.
2. Output JSON with three keys:
   - "hi_level1": The Hindi text strictly formatted in IndicVoices Level 1 (all English words transliterated into Devanagari script, all numbers written as Hindi words in Devanagari, disfluencies kept).
   - "en_literal": A literal word-by-word English translation that keeps any spoken self-correction (e.g., "fifteen lakh, no no, fifteen crore").
   - "en_fluent": A fluent English translation that resolves any self-correction to ONLY the final corrected value (e.g., "15 crore") so the abandoned value is not stated as fact.

Input Hindi: {hi_text}"""
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(resp.choices[0].message.content)
        if not manual_hi and data.get("hi_level1"):
            hi_text = enforce_indicvoices_level1(data["hi_level1"])
        return hi_text, data["en_literal"], data["en_fluent"]
    else:
        # Offline fallback if no API key is set
        return hi_text, "[Set GROQ_API_KEY or OPENAI_API_KEY for auto translation]", "[Offline mode]"

# ==============================================================================
# PART 4: TORCHAUDIO MMS FORCED ALIGNMENT (DEVANAGARI -> ROMAN -> CTC VITERBI)
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUNDLE = torchaudio.pipelines.MMS_FA
FA_MODEL = BUNDLE.get_model(with_star=False).to(DEVICE)
FA_MODEL.eval()
FA_DICT = BUNDLE.get_dict(star=None)

def romanize_word(word: str) -> str:
    """Converts a Devanagari word into lowercase ASCII characters supported by MMS_FA."""
    if UROMAN is not None:
        rom = UROMAN.romanize_string(word).lower()
    else:
        rom = "".join(DEV_TO_ROM.get(ch, "a") for ch in word)
    rom = re.sub(r"[^a-z]", "", rom)
    return rom if rom else "a"

def force_align_hindi(wav_path: Path, hi_text: str) -> Tuple[float, List[Dict]]:
    """
    Computes word-level [start, end] timestamps on 24 kHz audio using Meta MMS_FA.
    """
    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == 24000, f"Expected 24000 Hz audio, got {sr}"
    duration = round(len(audio_np) / sr, 3)

    # MMS_FA acoustic model expects 16 kHz input; timestamps map linearly to duration
    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(DEVICE)
    waveform_16k = torchaudio.functional.resample(waveform, sr, BUNDLE.sample_rate)

    raw_words = [w.strip("।,?!\"'.") for w in hi_text.split() if w.strip("।,?!\"'.")]
    rom_words = [romanize_word(w) for w in raw_words]

    with torch.inference_mode():
        emission, _ = FA_MODEL(waveform_16k)  # Shape: [1, num_frames, vocab_size]

    # Flatten character IDs across all words and track character length of each word
    flat_token_ids, word_char_lengths = [], []
    for rom in rom_words:
        char_ids = [FA_DICT[c] for c in rom if c in FA_DICT]
        if not char_ids:
            char_ids = [FA_DICT["a"]]
        flat_token_ids.extend(char_ids)
        word_char_lengths.append(len(char_ids))

    targets = torch.tensor([flat_token_ids], dtype=torch.int32, device=DEVICE)
    aligned_tokens, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
    char_spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])

    # Convert acoustic frame indices back to seconds
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

    return duration, hi_words

# ==============================================================================
# PART 5: PYTHON REGEX + NUMBER-TABLE ENTITY & NEGATION CHECKER
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

    # 1. Verify Final Corrected Amount
    for val, unit in amt_info["final"]:
        v_str = str(int(val)) if val == int(val) else str(val)
        pat = rf"\b{re.escape(v_str)}\s+{unit}s?\b"
        whole, thou = int(val), int(round((val - int(val)) * 100))
        alt_pat = rf"\b{whole}\s+{unit}s?\s+(?:and\s+)?{thou}\s+thousand\b"
        matched = bool(re.search(pat, en_norm) or (thou > 0 and re.search(alt_pat, en_norm)))
        results[f"final_amount:{v_str}_{unit}"] = (matched, f"Must contain corrected amount {v_str} {unit}")

    # 2. Check Self-Correction Integrity (Fluent must NOT state abandoned value as fact)
    if mode == "fluent" and amt_info["abandoned"]:
        for ab_val, ab_unit in amt_info["abandoned"]:
            if (ab_val, ab_unit) not in amt_info["final"]:
                ab_str = str(int(ab_val)) if ab_val == int(ab_val) else str(ab_val)
                leaked = bool(re.search(rf"\b{re.escape(ab_str)}\s+{ab_unit}s?\b", en_norm))
                results[f"self_corr_drop:{ab_str}_{ab_unit}"] = (not leaked, f"Fluent translation must drop abandoned {ab_str} {ab_unit}")

    # 3. Verify 10-Digit Phone Number
    en_digits = re.sub(r"\D", "", en_norm)
    for ph in extract_hi_phone(hi_text):
        results[f"phone:{ph}"] = (ph in en_digits, f"Must contain 10-digit phone {ph}")

    # 4. Verify Proper Names
    for hi_n, en_n in NAME_DICT.items():
        if hi_n in hi_text:
            results[f"name:{en_n}"] = (bool(re.search(rf"\b{en_n}\b", en_text.lower())), f"Must contain name '{en_n}'")

    # 5. Verify Negations (excluding 'नहीं नहीं' self-correction markers)
    cleaned_hi = re.sub("|".join(REPAIR_MARKERS), " ", hi_text)
    if {t.strip("।,?!.") for t in cleaned_hi.split()}.intersection(HI_NEGATION_WORDS):
        results["negation"] = (bool(EN_NEGATION_REGEX.search(en_text)), "Must preserve clause negation in English")

    return results

# ==============================================================================
# MAIN END-TO-END RUNNER FOR LIVE INTERVIEW
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Part D Pipeline for Live Interview")
    parser.add_argument("--audio", required=True, help="Path to ANY raw audio file (.m4a, .mp3, .wav)")
    parser.add_argument("--manual_hi", default=None, help="Optional manual Hindi transcript override")
    parser.add_argument("--manual_en_lit", default=None, help="Optional manual Literal English override")
    parser.add_argument("--manual_en_flu", default=None, help="Optional manual Fluent English override")
    args = parser.parse_args()

    in_path = Path(args.audio)
    clip_id = in_path.stem
    std_wav = Path(f"data/audio/{clip_id}_24k.wav")

    print(f"=== [1/5] Standardizing Audio -> 24 kHz, 16-bit Mono PCM ({std_wav}) ===")
    standardize_audio(in_path, std_wav)

    print("=== [2/5] Verbatim Level-1 Transcription & Dual Translation ===")
    if args.manual_hi and args.manual_en_lit and args.manual_en_flu:
        hi_text = enforce_indicvoices_level1(args.manual_hi)
        en_lit, en_flu = args.manual_en_lit, args.manual_en_flu
    else:
        hi_text, en_lit, en_flu = transcribe_and_translate_auto(std_wav, args.manual_hi)

    print(f"  Hindi (Level 1) : {hi_text}")
    print(f"  English Literal : {en_lit}")
    print(f"  English Fluent  : {en_flu}")

    print("=== [3/5] Running Torchaudio MMS Forced Alignment ===")
    duration, hi_words = force_align_hindi(std_wav, hi_text)
    label_file = Path(f"data/alignments/{clip_id}_audacity_labels.txt")
    label_file.parent.mkdir(parents=True, exist_ok=True)
    with open(label_file, "w", encoding="utf-8") as lf:
        for w in hi_words:
            lf.write(f"{w['start']:.3f}\t{w['end']:.3f}\t{w['w']}\n")
    print(f"  Aligned {len(hi_words)} words across {duration:.2f}s -> Saved Audacity labels to {label_file}")
    for w in hi_words[:8]:
        print(f"    [{w['start']:6.3f}s - {w['end']:6.3f}s] (conf={w['conf']:.2f}) -> {w['w']}")

    print("=== [4/5] Running Entity, Number, Name & Negation Checker ===")
    for label, text, m_type in [("LITERAL", en_lit, "literal"), ("FLUENT", en_flu, "fluent")]:
        print(f"  --- Checking {label} Translation ---")
        checks = verify_translation(hi_text, text, mode=m_type)
        for k, (ok, msg) in checks.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k:26s} | {msg}")

    print("=== [5/5] Exporting Section 5.2 Compliant JSONL Record ===")
    record = {
        "id": clip_id,
        "audio": str(std_wav).replace("\\", "/"),
        "duration": duration,
        "speaker_id": "spk-interview",
        "source": {"corpus": "live_eval", "source_file": str(in_path).replace("\\", "/"), "start": 0.0, "end": duration, "synthetic": False},
        "sentences": [{"hi_text": hi_text, "hi_start": hi_words[0]["start"], "hi_end": hi_words[-1]["end"], "en_text": en_lit, "en_alt": [en_flu]}],
        "hi_words": [{"w": x["w"], "start": x["start"], "end": x["end"]} for x in hi_words],
        "translation_method": "Automated ASR + LLM Dual-Target"
    }
    out_jsonl = Path(f"data/alignments/{clip_id}_manifest.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        jf.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved manifest -> {out_jsonl}")
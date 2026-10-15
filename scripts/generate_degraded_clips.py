#!/usr/bin/env python3
"""
scripts/generate_degraded_clips.py - Generate 5 Degraded Audio Test Conditions for Part E

Produces in data/part_e_audio/:
1. noisy_15dB.wav: Clean speech + multi-talker babble & fan hum at SNR = 15 dB.
2. noisy_10dB.wav: Clean speech + noise at SNR = 10 dB.
3. noisy_5dB.wav:  Clean speech + heavy noise at SNR = 5 dB.
4. tel_band_8k.wav: Telephone-band simulation (300-3400 Hz 4th-order Butterworth,
                   8 kHz downsampling, G.711 mu-law 256-level quantization, 24 kHz upsampling).
5. tel_noisy_10dB.wav: Telephone-band (300-3400 Hz @ 8 kHz) + 10 dB additive noise, upsampled to 24 kHz.
"""

import os
import glob
import numpy as np
import soundfile as sf
import scipy.signal as signal

def load_audio(file_path: str, target_sr: int = 24000) -> np.ndarray:
    data, sr = sf.read(file_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        num_target = int(round(len(data) * target_sr / sr))
        data = signal.resample(data, num_target)
    return data.astype(np.float32)

def generate_babble_noise(target_len: int, target_sr: int = 24000, raw_dir: str = "data/raw") -> np.ndarray:
    """Synthesizes realistic multi-talker babble from secondary clips."""
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
    # Exclude clip_1 to prevent self-leakage into noise
    noise_sources = [f for f in raw_files if "clip_1" not in os.path.basename(f)]
    
    babble = np.zeros(target_len, dtype=np.float32)
    np.random.seed(42)
    
    if noise_sources:
        for idx, src in enumerate(noise_sources):
            data = load_audio(src, target_sr=target_sr)
            # Repeat or tile to cover target_len
            if len(data) < target_len:
                reps = int(np.ceil(target_len / len(data)))
                data = np.tile(data, reps)
            # Random circular shift
            shift = np.random.randint(0, len(data))
            chunk = np.roll(data, shift)[:target_len]
            # Random scale
            chunk = chunk * np.random.uniform(0.4, 0.9)
            babble += chunk
        babble /= (len(noise_sources) + 1e-6)
    else:
        # Fallback if raw files not found: pink noise
        pink = np.random.randn(target_len).astype(np.float32)
        b, a = signal.butter(1, 0.1, btype='low')
        babble = signal.lfilter(b, a, pink)
        
    return babble

def generate_fan_hum_noise(target_len: int, target_sr: int = 24000) -> np.ndarray:
    """Synthesizes electrical hum (50 Hz + harmonics) and low-frequency HVAC rumble."""
    t = np.arange(target_len) / target_sr
    hum = (0.5 * np.sin(2 * np.pi * 50 * t) +
           0.3 * np.sin(2 * np.pi * 100 * t) +
           0.2 * np.sin(2 * np.pi * 150 * t) +
           0.1 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    
    # Add low-frequency HVAC rumble (pink noise lowpass filtered at 300 Hz)
    white = np.random.randn(target_len).astype(np.float32)
    b, a = signal.butter(2, 300 / (target_sr / 2), btype='low')
    rumble = signal.lfilter(b, a, white)
    
    ambient = 0.4 * hum + 0.6 * (rumble / (np.std(rumble) + 1e-8))
    return ambient.astype(np.float32)

def mix_at_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Mixes clean speech and noise at a target active-speech SNR (dB)."""
    # Active speech power estimation: threshold at -30 dB below peak RMS
    frame_len = 480
    rms_frames = np.array([np.sqrt(np.mean(clean[i:i+frame_len]**2) + 1e-12) 
                           for i in range(0, len(clean) - frame_len, frame_len)])
    peak_rms = np.max(rms_frames) if len(rms_frames) > 0 else 1.0
    active_thresh = 0.03 * peak_rms
    active_mask = np.repeat(rms_frames > active_thresh, frame_len)
    active_mask = np.pad(active_mask, (0, max(0, len(clean) - len(active_mask))), mode='constant')[:len(clean)]
    
    speech_power = np.mean(clean[active_mask]**2) if np.sum(active_mask) > 0 else np.mean(clean**2)
    noise_power = np.mean(noise**2) + 1e-12
    
    target_noise_power = speech_power / (10 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    
    mixed = clean + scale * noise
    # Prevent hard digital clipping while preserving dynamic range
    max_val = np.max(np.abs(mixed))
    if max_val > 0.98:
        mixed = mixed * (0.98 / max_val)
    return mixed.astype(np.float32)

def mu_law_quantize(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Standard ITU-T G.711 mu-law compression, 8-bit uniform quantization, expansion."""
    x = np.clip(x, -1.0, 1.0)
    # Compression: F(x) = sgn(x) * ln(1 + mu*|x|) / ln(1 + mu)
    comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # Quantize to 256 levels in [-1, 1]
    quant_int = np.round(((comp + 1.0) / 2.0) * 255.0)
    quant_int = np.clip(quant_int, 0, 255)
    # Dequantize
    dequant = (quant_int / 255.0) * 2.0 - 1.0
    # Expansion: F^-1(y) = sgn(y) * ((1 + mu)^|y| - 1) / mu
    expanded = np.sign(dequant) * (np.power(1.0 + mu, np.abs(dequant)) - 1.0) / mu
    return expanded.astype(np.float32)

def simulate_telephone_band(clean_24k: np.ndarray, add_noise_snr: float = None) -> np.ndarray:
    """
    Simulates standard PSTN telephone line:
    1. 4th-order Butterworth bandpass filter [300 Hz, 3400 Hz] @ 24 kHz
    2. Downsample to 8000 Hz
    3. G.711 mu-law 8-bit quantization distortion
    4. Optional additive noise at 8 kHz
    5. Upsample back to 24000 Hz
    """
    fs_in = 24000
    fs_tel = 8000
    
    # 4th-order Butterworth bandpass [300 Hz, 3400 Hz]
    b, a = signal.butter(4, [300 / (fs_in / 2), 3400 / (fs_in / 2)], btype='bandpass')
    filtered = signal.lfilter(b, a, clean_24k)
    
    # Downsample from 24 kHz to 8 kHz (factor of 3)
    down_8k = signal.resample_poly(filtered, up=1, down=3)
    
    # G.711 mu-law compression + 8-bit quantization
    quant_8k = mu_law_quantize(down_8k, mu=255)
    
    if add_noise_snr is not None:
        noise_8k = np.random.randn(len(quant_8k)).astype(np.float32)
        # bandpass the noise to telephone bandwidth
        b_n, a_n = signal.butter(2, [300 / (fs_tel / 2), 3400 / (fs_tel / 2)], btype='bandpass')
        noise_8k = signal.lfilter(b_n, a_n, noise_8k)
        quant_8k = mix_at_snr(quant_8k, noise_8k, add_noise_snr)
        
    # Upsample back to 24 kHz (factor of 3)
    up_24k = signal.resample_poly(quant_8k, up=3, down=1)
    
    # Match length to clean_24k
    if len(up_24k) < len(clean_24k):
        up_24k = np.pad(up_24k, (0, len(clean_24k) - len(up_24k)), mode='constant')
    else:
        up_24k = up_24k[:len(clean_24k)]
        
    return up_24k.astype(np.float32)

def main():
    clean_path = "data/audio/clip_1_24k.wav"
    out_dir = "data/part_e_audio"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Clean reference audio not found at: {clean_path}")
        
    clean_audio = load_audio(clean_path, target_sr=24000)
    target_len = len(clean_audio)
    print(f"Loaded clean reference: {clean_path} ({len(clean_audio)} samples, {len(clean_audio)/24000:.2f}s)")
    
    # Generate realistic ambient composite noise (70% babble + 30% fan/hum)
    print("Synthesizing realistic non-stationary multi-talker babble & fan/hum noise...")
    babble = generate_babble_noise(target_len, target_sr=24000)
    hum = generate_fan_hum_noise(target_len, target_sr=24000)
    composite_noise = 0.70 * (babble / (np.std(babble) + 1e-8)) + 0.30 * (hum / (np.std(hum) + 1e-8))
    composite_noise = composite_noise.astype(np.float32)
    
    # Condition 1: Noisy 15 dB
    noisy_15dB = mix_at_snr(clean_audio, composite_noise, target_snr_db=15.0)
    path_15 = os.path.join(out_dir, "noisy_15dB.wav")
    sf.write(path_15, noisy_15dB, 24000, subtype="PCM_16")
    print(f"  [1/5] Generated {path_15} (SNR = 15 dB)")
    
    # Condition 2: Noisy 10 dB
    noisy_10dB = mix_at_snr(clean_audio, composite_noise, target_snr_db=10.0)
    path_10 = os.path.join(out_dir, "noisy_10dB.wav")
    sf.write(path_10, noisy_10dB, 24000, subtype="PCM_16")
    print(f"  [2/5] Generated {path_10} (SNR = 10 dB)")
    
    # Condition 3: Noisy 5 dB
    noisy_5dB = mix_at_snr(clean_audio, composite_noise, target_snr_db=5.0)
    path_5 = os.path.join(out_dir, "noisy_5dB.wav")
    sf.write(path_5, noisy_5dB, 24000, subtype="PCM_16")
    print(f"  [3/5] Generated {path_5} (SNR = 5 dB)")
    
    # Condition 4: Telephone-band 8k simulation
    tel_band_8k = simulate_telephone_band(clean_audio, add_noise_snr=None)
    path_tel = os.path.join(out_dir, "tel_band_8k.wav")
    sf.write(path_tel, tel_band_8k, 24000, subtype="PCM_16")
    print(f"  [4/5] Generated {path_tel} (PSTN 300-3400 Hz, 8 kHz mu-law quantized)")
    
    # Condition 5: Telephone-band with 10 dB additive noise
    tel_noisy_10dB = simulate_telephone_band(clean_audio, add_noise_snr=10.0)
    path_tel_noisy = os.path.join(out_dir, "tel_noisy_10dB.wav")
    sf.write(path_tel_noisy, tel_noisy_10dB, 24000, subtype="PCM_16")
    print(f"  [5/5] Generated {path_tel_noisy} (Telephone-band + 10 dB noise)")
    
    print("\nAll 5 degraded conditions successfully generated in data/part_e_audio/!")

if __name__ == "__main__":
    main()

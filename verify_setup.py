import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import torchaudio
import soundfile as sf
import uroman

print("=" * 40)
print("DEVICE & HARDWARE VERIFICATION")
print("=" * 40)
print(f"Python Executable: {sys.executable}")
print(f"Python Version   : {sys.version.split()[0]}")
print(f"Torch Version    : {torch.__version__}")
print(f"CUDA Available   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Count     : {torch.cuda.device_count()}")
    print(f"Device Name      : {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute Cap.     : {major}.{minor}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total VRAM       : {vram_gb:.2f} GB")
    # Quick tensor test
    t = torch.randn(1000, 1000, device="cuda")
    matmul = torch.matmul(t, t)
    print(f"GPU MatMul Test  : PASSED (Shape: {matmul.shape}, Device: {matmul.device})")
else:
    print("WARNING: CUDA is not available to Torch.")

print("\n" + "=" * 40)
print("AUDIO & NLP PACKAGES")
print("=" * 40)
print(f"Torchaudio Version: {torchaudio.__version__}")
print(f"SoundFile Version : {sf.__version__}")

# SoundFile test
import numpy as np
test_audio = np.random.uniform(-1.0, 1.0, size=(16000,)).astype(np.float32)
sf.write("test_audio_sample.wav", test_audio, 16000)
data, samplerate = sf.read("test_audio_sample.wav")
print(f"SoundFile I/O Test: PASSED (Sample Rate: {samplerate}Hz, Data points: {len(data)})")
import os
os.remove("test_audio_sample.wav")

# Uroman test
test_text = "नमस्ते दुनिया"
romanized = uroman.uroman(test_text)
print(f"uroman Romanization: PASSED ('{test_text}' -> '{romanized}')")

print("\n" + "=" * 40)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("=" * 40)

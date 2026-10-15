#!/usr/bin/env python3
"""
scripts/run_live_interview.py - Live Interview Execution Pipeline for Inter IIT Tech Meet 15.0

Supports:
- Part D: ASR Transcription (Whisper-large-v3-turbo), MMS Forced Alignment (sub-20ms precision),
  Level-1 Hindi Normalization, 7B Dual Translation, and Universal Entity Verification.
- Part E: Strictly causal 80 ms audio front-end (resampling, denoising, Silero VAD) enabled via
  --run_sp_frontend flag.
"""

import sys
import os
import subprocess

def get_best_python():
    """Detects .venv python if current environment is missing audio packages."""
    try:
        import torch, soundfile  # noqa: F401
        return sys.executable
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
        candidates = [
            os.path.join(workspace_root, ".venv", "Scripts", "python.exe"),
            os.path.join(workspace_root, ".venv", "bin", "python"),
            os.path.join(workspace_root, "venv", "Scripts", "python.exe"),
            os.path.join(workspace_root, "venv", "bin", "python"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return sys.executable

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_py = os.path.join(script_dir, "run_interview_pipeline.py")
    py_exec = get_best_python()
    
    cmd = [py_exec, pipeline_py] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()


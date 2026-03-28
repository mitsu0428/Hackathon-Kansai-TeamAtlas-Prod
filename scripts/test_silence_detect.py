"""Test if _is_silence works correctly.

Run: backend/.venv/bin/python scripts/test_silence_detect.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.infra.clap_model import ClapModel

print("Loading CLAP...")
clap = ClapModel()
clap.load()

# Test embed_text
print("\n=== embed_text test ===")
try:
    v = clap.embed_text("silence")
    print(f"embed_text OK: shape={v.shape}")
except Exception as e:
    print(f"embed_text FAILED: {e}")
    sys.exit(1)

# Test silence similarity
print("\n=== silence similarity ===")
texts = ["silence", "no sound", "quiet"]
vecs = [clap.embed_text(t) for t in texts]
avg = np.mean(vecs, axis=0).astype(np.float32)
norm = np.linalg.norm(avg)
silence_vec = avg / norm if norm > 0 else avg

audio_silence = np.zeros(48000 * 3, dtype=np.float32)
emb = clap.embed(audio_silence)
emb_norm = emb / max(np.linalg.norm(emb), 1e-8)
sim = float(np.dot(emb_norm, silence_vec))
print(f"true silence sim: {sim:.4f} (threshold=0.5, should be > 0.5)")

audio_noise = 0.005 * np.random.randn(48000 * 3).astype(np.float32)
emb2 = clap.embed(audio_noise)
emb2_norm = emb2 / max(np.linalg.norm(emb2), 1e-8)
sim2 = float(np.dot(emb2_norm, silence_vec))
print(f"near silence sim: {sim2:.4f}")

# Test _is_silence function directly
print("\n=== _is_silence function test ===")
from src.use_case.detect_anomaly import _is_silence

result = _is_silence(clap, emb)
print(f"_is_silence(true_silence): {result}")

result2 = _is_silence(clap, emb2)
print(f"_is_silence(near_silence): {result2}")

print("\nDone.")

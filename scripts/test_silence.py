"""Test silence detection vs normal audio detection.

Run on GPU server: backend/.venv/bin/python scripts/test_silence.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.infra.audio_loader import load_audio
from src.infra.clap_model import ClapModel
from src.infra.config import settings
from src.infra.faiss_index import FaissIndex


def main():
    print("=== Silence Detection Test ===\n")

    # Load CLAP
    print("Loading CLAP model...")
    clap = ClapModel()
    clap.load()

    # Load indoor index
    idx_path = settings.index_dir / "indoor_baseline.faiss"
    meta_path = settings.index_dir / "indoor_metadata.jsonl"
    print(f"Loading indoor index from {idx_path}...")
    index = FaissIndex(512)
    index.load(str(idx_path))
    if meta_path.exists():
        index.load_metadata(str(meta_path))
    print(f"Index loaded: {index.ntotal} vectors\n")

    # Test 1: True silence (all zeros)
    print("--- Test 1: True silence (all zeros) ---")
    silence = np.zeros(48000 * 3, dtype=np.float32)
    emb = clap.embed(silence)
    sims, _, _ = index.search(emb, k=5)
    best = float(np.max(sims[0]))
    dist = max(0.0, 1.0 - best)
    print(f"  best_sim={best:.4f} distance={dist:.4f}\n")

    # Test 2: Near silence (tiny noise, like _generate_audio("silence"))
    print("--- Test 2: Near silence (tiny noise 0.005) ---")
    near_silence = 0.005 * np.random.randn(48000 * 3).astype(np.float32)
    emb = clap.embed(near_silence)
    sims, _, _ = index.search(emb, k=5)
    best = float(np.max(sims[0]))
    dist = max(0.0, 1.0 - best)
    print(f"  best_sim={best:.4f} distance={dist:.4f}\n")

    # Test 3: Real stream audio (first file from indoor)
    print("--- Test 3: Real indoor stream audio ---")
    stream_dir = settings.streams_dir / "indoor"
    stream_files = sorted(stream_dir.glob("*.wav"))[:3]
    for f in stream_files:
        audio = load_audio(f)
        emb = clap.embed(audio)
        sims, _, _ = index.search(emb, k=5)
        best = float(np.max(sims[0]))
        dist = max(0.0, 1.0 - best)
        print(f"  {f.name}: best_sim={best:.4f} distance={dist:.4f}")
    print()

    # Test 4: Generated "normal" audio (sine wave)
    print("--- Test 4: Generated sine wave (fake normal) ---")
    t = np.linspace(0, 3.0, 48000 * 3)
    sine = (0.05 * np.random.randn(len(t)) + 0.15 * np.sin(2 * np.pi * 120 * t)).astype(
        np.float32
    )
    emb = clap.embed(sine)
    sims, _, _ = index.search(emb, k=5)
    best = float(np.max(sims[0]))
    dist = max(0.0, 1.0 - best)
    print(f"  best_sim={best:.4f} distance={dist:.4f}\n")

    # Test 5: CLAP text encoder (if available)
    print("--- Test 5: CLAP text similarity ---")
    try:
        silence_emb = clap.embed_text("silence")
        silence_emb = silence_emb / np.linalg.norm(silence_emb)

        # Silence audio vs "silence" text
        s_emb = clap.embed(silence)
        s_emb = s_emb / np.linalg.norm(s_emb)
        print(f"  true_silence vs 'silence' text: {float(np.dot(s_emb, silence_emb)):.4f}")

        # Near silence vs "silence" text
        ns_emb = clap.embed(near_silence)
        ns_emb = ns_emb / np.linalg.norm(ns_emb)
        print(f"  near_silence vs 'silence' text: {float(np.dot(ns_emb, silence_emb)):.4f}")

        # Real audio vs "silence" text
        if stream_files:
            real = load_audio(stream_files[0])
            r_emb = clap.embed(real)
            r_emb = r_emb / np.linalg.norm(r_emb)
            print(f"  real_audio vs 'silence' text:  {float(np.dot(r_emb, silence_emb)):.4f}")
    except Exception as e:
        print(f"  Text encoder not available: {e}")

    print("\n=== Done ===")
    print(f"Threshold: {settings.ANOMALY_THRESHOLD}")
    print("If silence distance < threshold, silence detection is broken.")


if __name__ == "__main__":
    main()

"""
Pet Communicator — Audio Feature Extractor

Usage:
    python analyze_audio.py <audio_file_path>

Output: JSON to stdout with audio features for downstream interpretation.

Strategy:
  1. Try librosa (best quality) if available.
  2. Fall back to scipy.io.wavfile + numpy.
  3. Fall back to wave + numpy (stdlib).
  4. If file is not wav and librosa absent, attempt ffmpeg conversion to a temp wav.

The script never raises on missing optional libs; it degrades gracefully and
includes a "backend" field so the caller knows how the features were derived.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _try_import_librosa():
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
        return librosa, np
    except Exception:
        return None, None


def _try_import_scipy_numpy():
    try:
        import numpy as np  # type: ignore
        from scipy.io import wavfile  # type: ignore
        return np, wavfile
    except Exception:
        try:
            import numpy as np  # type: ignore
            return np, None
        except Exception:
            return None, None


def _ffmpeg_to_wav(src: Path) -> Path | None:
    if shutil.which("ffmpeg") is None:
        return None
    tmp = Path(tempfile.gettempdir()) / f"pet_communicator_{os.getpid()}.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src),
                "-ac", "1", "-ar", "22050",
                str(tmp),
            ],
            check=True,
        )
        return tmp
    except Exception:
        return None


def analyze_with_librosa(path: Path) -> dict[str, Any]:
    import librosa  # type: ignore
    import numpy as np  # type: ignore

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = float(len(y) / sr) if sr else 0.0

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_max = float(np.max(rms)) if rms.size else 0.0

    # Pitch via piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    voiced_pitches = []
    for t in range(pitches.shape[1]):
        idx = int(magnitudes[:, t].argmax())
        p = float(pitches[idx, t])
        if p > 50:
            voiced_pitches.append(p)
    pitch_arr = np.array(voiced_pitches) if voiced_pitches else np.array([])
    pitch_mean = float(pitch_arr.mean()) if pitch_arr.size else 0.0
    pitch_std = float(pitch_arr.std()) if pitch_arr.size else 0.0
    pitch_range = float(pitch_arr.max() - pitch_arr.min()) if pitch_arr.size else 0.0

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_mean = float(np.mean(spectral_centroid)) if spectral_centroid.size else 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(tempo) if tempo else None
    except Exception:
        tempo_bpm = None

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    onset_count = int(len(onset_frames))

    # voiced ratio: frames with rms > 10% of max
    if rms.size:
        thr = rms_max * 0.1
        voiced_ratio = float(np.mean(rms > thr))
    else:
        voiced_ratio = 0.0

    return {
        "backend": "librosa",
        "duration_sec": round(duration, 3),
        "rms_mean": round(rms_mean, 5),
        "rms_max": round(rms_max, 5),
        "pitch_hz_mean": round(pitch_mean, 1),
        "pitch_hz_std": round(pitch_std, 1),
        "pitch_hz_range": round(pitch_range, 1),
        "spectral_centroid_mean": round(sc_mean, 1),
        "zero_crossing_rate_mean": round(zcr_mean, 4),
        "tempo_bpm": round(tempo_bpm, 1) if tempo_bpm else None,
        "onset_count": onset_count,
        "voiced_ratio": round(voiced_ratio, 3),
        "sample_rate": int(sr),
    }


def analyze_with_scipy(path: Path) -> dict[str, Any]:
    import numpy as np  # type: ignore
    from scipy.io import wavfile  # type: ignore

    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.size:
        peak = float(np.max(np.abs(data))) or 1.0
        data = data / peak

    duration = float(len(data) / sr) if sr else 0.0

    # frame-based RMS
    frame = max(int(sr * 0.025), 1)
    hop = max(int(sr * 0.01), 1)
    rms_vals = []
    for i in range(0, max(0, len(data) - frame), hop):
        chunk = data[i : i + frame]
        rms_vals.append(float(np.sqrt(np.mean(chunk ** 2))))
    rms_arr = np.array(rms_vals) if rms_vals else np.array([0.0])
    rms_mean = float(rms_arr.mean())
    rms_max = float(rms_arr.max())

    # Dominant frequency via FFT on full signal (rough)
    n = len(data)
    if n > 0:
        fft = np.fft.rfft(data * np.hanning(n))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mag = np.abs(fft)
        # Look only in 50–8000 Hz
        mask = (freqs >= 50) & (freqs <= 8000)
        if mask.any():
            mag_band = mag[mask]
            freqs_band = freqs[mask]
            dominant = float(freqs_band[int(mag_band.argmax())])
            # weighted mean as "centroid"
            sc = float(np.sum(freqs_band * mag_band) / (np.sum(mag_band) + 1e-9))
        else:
            dominant = 0.0
            sc = 0.0
    else:
        dominant = 0.0
        sc = 0.0

    # Zero-crossing rate
    zc = np.sum(np.abs(np.diff(np.sign(data)))) / 2 if data.size else 0
    zcr_mean = float(zc / len(data)) if len(data) else 0.0

    # Onset: count rms peaks above threshold
    thr = rms_max * 0.4
    onsets = 0
    above = False
    for v in rms_vals:
        if v >= thr and not above:
            onsets += 1
            above = True
        elif v < thr * 0.5:
            above = False

    voiced_ratio = float(np.mean(rms_arr > rms_max * 0.1)) if rms_arr.size else 0.0

    return {
        "backend": "scipy",
        "duration_sec": round(duration, 3),
        "rms_mean": round(rms_mean, 5),
        "rms_max": round(rms_max, 5),
        "pitch_hz_mean": round(dominant, 1),
        "pitch_hz_std": None,
        "pitch_hz_range": None,
        "spectral_centroid_mean": round(sc, 1),
        "zero_crossing_rate_mean": round(zcr_mean, 4),
        "tempo_bpm": None,
        "onset_count": int(onsets),
        "voiced_ratio": round(voiced_ratio, 3),
        "sample_rate": int(sr),
    }


def analyze_with_wave(path: Path) -> dict[str, Any]:
    import wave
    import numpy as np  # type: ignore

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_frames = w.getnframes()
        sw = w.getsampwidth()
        n_channels = w.getnchannels()
        raw = w.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    if data.size:
        peak = float(np.max(np.abs(data))) or 1.0
        data = data / peak

    duration = float(len(data) / sr) if sr else 0.0
    rms_mean = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
    rms_max = float(np.max(np.abs(data))) if data.size else 0.0

    return {
        "backend": "wave",
        "duration_sec": round(duration, 3),
        "rms_mean": round(rms_mean, 5),
        "rms_max": round(rms_max, 5),
        "pitch_hz_mean": None,
        "pitch_hz_std": None,
        "pitch_hz_range": None,
        "spectral_centroid_mean": None,
        "zero_crossing_rate_mean": None,
        "tempo_bpm": None,
        "onset_count": None,
        "voiced_ratio": None,
        "sample_rate": int(sr),
    }


def build_notes(features: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    dur = features.get("duration_sec") or 0
    rms_mean = features.get("rms_mean") or 0
    pitch = features.get("pitch_hz_mean") or 0
    pitch_std = features.get("pitch_hz_std")
    onsets = features.get("onset_count")
    sc = features.get("spectral_centroid_mean") or 0
    zcr = features.get("zero_crossing_rate_mean") or 0

    # Duration band
    if dur < 0.3:
        notes.append("音檔極短（< 0.3 秒），特徵代表性偏低")
    elif dur < 1.0:
        notes.append("短促叫聲")
    elif dur < 3.0:
        notes.append("中等長度叫聲")
    else:
        notes.append("拉長/持續性叫聲")

    # Volume
    if rms_mean < 0.02:
        notes.append("音量微弱（可能距離遠或刻意壓低）")
    elif rms_mean > 0.2:
        notes.append("音量響亮")

    # Pitch
    if pitch:
        if pitch < 200:
            notes.append(f"低頻為主（約 {pitch:.0f} Hz），偏低沉")
        elif pitch < 800:
            notes.append(f"中頻為主（約 {pitch:.0f} Hz）")
        elif pitch < 2000:
            notes.append(f"中高頻為主（約 {pitch:.0f} Hz）")
        else:
            notes.append(f"高頻尖銳（約 {pitch:.0f} Hz）")

    # Pitch variability
    if pitch_std is not None:
        if pitch_std < 50:
            notes.append("音高平穩")
        elif pitch_std < 200:
            notes.append("音高有起伏")
        else:
            notes.append("音高劇烈起伏")

    # Onsets
    if onsets is not None:
        if onsets <= 1:
            notes.append("單聲事件")
        elif onsets <= 4:
            notes.append(f"重複數次（約 {onsets} 聲）")
        else:
            notes.append(f"高密度重複（約 {onsets} 聲）")

    # Brightness via spectral centroid
    if sc:
        if sc < 800:
            notes.append("音色偏暗沉/低沉")
        elif sc > 3000:
            notes.append("音色尖銳/明亮")

    # ZCR
    if zcr and zcr > 0.15:
        notes.append("含明顯擦音/嘶聲成分")

    return notes


def analyze(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"error": f"file not found: {path}"}

    librosa, _ = _try_import_librosa()
    if librosa is not None:
        try:
            feats = analyze_with_librosa(path)
            feats["notes"] = build_notes(feats)
            return feats
        except Exception as e:
            err_librosa = str(e)
    else:
        err_librosa = "librosa not installed"

    # Need a wav for the fallbacks
    wav_path = path
    if path.suffix.lower() != ".wav":
        converted = _ffmpeg_to_wav(path)
        if converted is None:
            return {
                "error": "non-wav file and librosa unavailable; install librosa or ensure ffmpeg is on PATH",
                "librosa_error": err_librosa,
            }
        wav_path = converted

    np_, wavfile = _try_import_scipy_numpy()
    if np_ is not None and wavfile is not None:
        try:
            feats = analyze_with_scipy(wav_path)
            feats["notes"] = build_notes(feats)
            return feats
        except Exception as e:
            err_scipy = str(e)
    else:
        err_scipy = "scipy or numpy not installed"

    if np_ is not None:
        try:
            feats = analyze_with_wave(wav_path)
            feats["notes"] = build_notes(feats)
            feats["degraded"] = True
            return feats
        except Exception as e:
            return {
                "error": "all backends failed",
                "librosa_error": err_librosa,
                "scipy_error": err_scipy,
                "wave_error": str(e),
            }

    return {
        "error": "numpy not installed; please run: pip install librosa numpy scipy",
        "librosa_error": err_librosa,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: analyze_audio.py <audio_path>"}))
        return 1
    path = Path(sys.argv[1]).expanduser()
    result = analyze(path)
    result["input_path"] = str(path)
    result["filename"] = path.name
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 2


if __name__ == "__main__":
    raise SystemExit(main())

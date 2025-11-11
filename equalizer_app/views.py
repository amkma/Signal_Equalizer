# equalizer_app/views.py
import io, json, os, uuid, wave, math, struct, base64
from dataclasses import dataclass
from typing import List, Dict
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import numpy as np  # allowed for arrays, not for fft/spectrogram
from .utils.fft import fft_radix2, ifft_radix2, stft_spectrogram

MEDIA_ROOT = getattr(settings, "MEDIA_ROOT", os.path.join(settings.BASE_DIR, "media"))
DATA_DIR = os.path.join(MEDIA_ROOT, "signals")
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory light registry (you can move to DB if needed)
REG: Dict[str, dict] = {}


def _resp_json(data: dict):
    buf = json.dumps(data).encode("utf-8")
    return HttpResponse(buf, content_type="application/json")


def _write_wav(path, sr, x: np.ndarray):
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sr))
        # PCM 16
        frames = (x * 32767.0).astype(np.int16).tobytes()
        wf.writeframes(frames)


def _read_wav(fileobj):
    with wave.open(fileobj, "rb") as wf:
        nchan = wf.getnchannels()
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nchan > 1:
        x = x.reshape(-1, nchan).mean(axis=1)
    return sr, x


def _downsample_preview(x: np.ndarray, target=2000):
    if x.size <= target:
        return x.tolist()
    idx = np.linspace(0, x.size - 1, target).astype(np.int64)
    return x[idx].tolist()


def _next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p


def _make_output_for_signal(sid):
    meta = REG[sid]
    in_x = meta["input_x"]
    sr = meta["sr"]
    # if no settings yet, output == input
    out_x = meta.get("output_x", in_x.copy())
    out_path = os.path.join(DATA_DIR, sid, "output.wav")
    _write_wav(out_path, sr, out_x)
    REG[sid]["output_x"] = out_x
    return out_path


@csrf_exempt
def upload_signal(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST only")
    f = request.FILES.get("signal")
    if not f:
        return HttpResponseBadRequest("No file 'signal'")
    sid = uuid.uuid4().hex[:12]
    sig_dir = os.path.join(DATA_DIR, sid)
    os.makedirs(sig_dir, exist_ok=True)

    # store original
    orig_path = os.path.join(sig_dir, f.name)
    with open(orig_path, "wb") as fp:
        for chunk in f.chunks():
            fp.write(chunk)

    # load wav -> sr, x
    with open(orig_path, "rb") as fp:
        sr, x = _read_wav(fp)

    REG[sid] = {
        "file_name": f.name,
        "sr": int(sr),
        "input_x": x.astype(np.float32),
        "output_x": x.astype(np.float32).copy(),
        "mode": "generic",
        "subbands": [],
        "custom_sliders": [],
        "scale": "linear",
        "show_spec": True,
    }
    _make_output_for_signal(sid)

    resp = {
        "signal_id": sid,
        "file_name": f.name,
        "sr": int(sr),
        "n": int(x.size),
        "duration": float(x.size / sr),
    }
    return _resp_json(resp)


def summary(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    resp = {
        "file_name": meta["file_name"],
        "sr": meta["sr"],
        "duration": float(meta["input_x"].size / meta["sr"]),
    }
    return _resp_json(resp)


# ---
# --- THIS FUNCTION IS NOW OPTIMIZED ---
# ---
def _compute_spectrum(x: np.ndarray, sr: int, scale: str):
    """
    Calculates the *average* magnitude spectrum.
    This is much faster and smoother than an FFT of the whole file.
    """
    # S is a 2D array: (frequency_bins, time_frames)
    S = stft_spectrogram(x, sr)

    if S.size == 0:
        # Handle very short files that produce no frames
        nfft = 1
        while nfft < int(sr * 25 / 1000): nfft <<= 1  # Recreate default nfft
        return [0.0] * (nfft // 2), sr / 2.0

    # The "average spectrum" is the mean across all time frames
    mag = np.mean(S, axis=1)

    # Normalize
    mag_max = mag.max()
    if mag_max > 1e-12:
        mag = mag / mag_max

    fmax = sr / 2.0

    if scale == "audiogram":
        # Emphasize speech band for visualization
        mag = np.power(mag, 0.7)

    return mag.tolist(), float(fmax)


def spectrum(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    scale = request.GET.get("scale", "linear")
    # Get output_x, fall back to input_x
    x_data = meta.get("output_x", meta["input_x"])
    mags, fmax = _compute_spectrum(x_data, meta["sr"], scale)
    return _resp_json({"mags": mags, "fmax": fmax})


def wave_previews(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    inp = _downsample_preview(meta["input_x"])
    out = _downsample_preview(meta.get("output_x", meta["input_x"]))
    return _resp_json({"input": inp, "output": out})


def spectrograms(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    sr = meta["sr"]
    x_in = meta["input_x"]
    x_out = meta.get("output_x", x_in)
    S_in = stft_spectrogram(x_in, sr)
    S_out = stft_spectrogram(x_out, sr)

    # convert to PNG base64 (simple grayscale)
    import PIL.Image as Image  # Pillow is okay for rendering images; not part of FFT
    def to_png_b64(S):
        # normalize
        Sm = S - S.min()
        s_max = Sm.max()
        if s_max < 1e-12:
            Sm = np.zeros_like(Sm)
        else:
            Sm = Sm / s_max

        img = (Sm * 255).astype(np.uint8)
        im = Image.fromarray(img[::-1, :], mode="L")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return _resp_json({"in_png": to_png_b64(S_in), "out_png": to_png_b64(S_out)})


def custom_conf(request, sid):
    """Return predefined sliders for the chosen mode."""
    mode = request.GET.get("mode", "generic").lower()
    sliders = []
    if mode == "musical instruments":
        sliders = [
            {"name": "Piano", "gain": 1.0, "windows": [{"fmin": 150, "fmax": 1200}, {"fmin": 2000, "fmax": 4000}]},
            {"name": "Drums", "gain": 1.0, "windows": [{"fmin": 40, "fmax": 180}]},
            {"name": "Violin", "gain": 1.0, "windows": [{"fmin": 300, "fmax": 3500}]},
            {"name": "Bass", "gain": 1.0, "windows": [{"fmin": 40, "fmax": 200}]},
        ]
    elif mode == "animal sounds":
        sliders = [
            {"name": "Dog", "gain": 1.0, "windows": [{"fmin": 400, "fmax": 2000}]},
            {"name": "Cat", "gain": 1.0, "windows": [{"fmin": 500, "fmax": 3000}]},
            {"name": "Horse", "gain": 1.0, "windows": [{"fmin": 100, "fmax": 800}]},
            {"name": "Bird", "gain": 1.0, "windows": [{"fmin": 2000, "fmax": 7000}]},
        ]
    elif mode == "human voices":
        sliders = [
            {"name": "Male", "gain": 1.0, "windows": [{"fmin": 85, "fmax": 180}, {"fmin": 2000, "fmax": 4000}]},
            {"name": "Female", "gain": 1.0, "windows": [{"fmin": 165, "fmax": 255}, {"fmin": 2500, "fmax": 5000}]},
            {"name": "Child", "gain": 1.0, "windows": [{"fmin": 250, "fmax": 400}, {"fmin": 3000, "fmax": 6000}]},
            {"name": "Sibilants", "gain": 1.0, "windows": [{"fmin": 4000, "fmax": 10000}]},
        ]
    return _resp_json({"sliders": sliders})


@csrf_exempt
def equalize(request, sid):
    if request.method != "POST": return HttpResponseBadRequest("POST only")
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    body = json.loads(request.body.decode("utf-8"))
    mode = body.get("mode", "generic")
    sr = meta["sr"]
    x = meta["input_x"].astype(np.float32)

    # FFT-based scaling in frequency domain (manual FFT)
    n = x.size
    n2 = _next_pow2(n)
    xz = np.zeros(n2, dtype=np.float32);
    xz[:n] = x
    X = fft_radix2(xz)

    def apply_windows(windows, gain):
        # windows: list of {fmin,fmax}
        for w in windows:
            fmin = max(0.0, float(w["fmin"]))
            fmax = max(0.0, float(w["fmax"]))
            if fmax < fmin: fmin, fmax = fmax, fmin
            kmin = int(fmin / (sr / n2))
            kmax = int(fmax / (sr / n2))
            kmin = max(0, min(kmin, n2 // 2 - 1))
            kmax = max(0, min(kmax, n2 // 2 - 1))
            if kmax < kmin:
                continue
            # mirror bins for Hermitian symmetry (real signal)
            X[kmin:kmax + 1] *= gain
            if kmin > 0:
                X[-(kmax + 1):-(kmin)].__imul__(gain)

    if mode == "generic":
        subs = body.get("subbands", [])
        REG[sid]["subbands"] = subs
        for sb in subs:
            apply_windows([{"fmin": sb["fmin"], "fmax": sb["fmax"]}], float(sb["gain"]))
    else:
        sliders = body.get("sliders", [])
        REG[sid]["custom_sliders"] = sliders
        for s in sliders:
            g = float(s.get("gain", 1.0))
            wins = s.get("windows", [])
            apply_windows(wins, g)

    xr = ifft_radix2(X).real[:n].astype(np.float32)
    REG[sid]["output_x"] = xr
    _make_output_for_signal(sid)
    return _resp_json({"ok": True})


@csrf_exempt
def save_scheme(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    body = json.loads(request.body.decode("utf-8"))
    name = f"scheme_{sid}.json"
    return _resp_json({"filename": name, "data": body})


@csrf_exempt
def load_scheme(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    body = json.loads(request.body.decode("utf-8"))
    if body.get("mode", "generic") == "generic":
        REG[sid]["mode"] = "generic"
        REG[sid]["subbands"] = body.get("subbands", [])
    else:
        REG[sid]["mode"] = body.get("mode")
        REG[sid]["custom_sliders"] = body.get("sliders", [])
    return _resp_json({"ok": True})


@csrf_exempt
def save_settings(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    body = json.loads(request.body.decode("utf-8"))
    name = f"settings_{sid}.json"
    return _resp_json({"filename": name, "data": body})


@csrf_exempt
def load_settings(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    body = json.loads(request.body.decode("utf-8"))
    REG[sid]["scale"] = body.get("scale", "linear")
    REG[sid]["show_spec"] = bool(body.get("showSpectrograms", True))
    if body.get("mode", "generic") == "generic":
        REG[sid]["mode"] = "generic"
        REG[sid]["subbands"] = body.get("subbands", [])
    else:
        REG[sid]["mode"] = body.get("mode")
        REG[sid]["custom_sliders"] = body.get("sliders", [])
    return _resp_json({"ok": True})


def audio_input(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    sr = meta["sr"];
    x = meta["input_x"]
    buf = io.BytesIO()
    _write_wav(buf, sr, x)
    return HttpResponse(buf.getvalue(), content_type="audio/wav")


def audio_output(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    sr = meta["sr"];
    x = meta.get("output_x", meta["input_x"])
    buf = io.BytesIO()
    _write_wav(buf, sr, x)
    return HttpResponse(buf.getvalue(), content_type="audio/wav")


@csrf_exempt
def ai_run(request, sid):
    # Placeholder: Return fake stems+sliders; you can hook real models later
    body = json.loads(request.body.decode("utf-8"))
    model = body.get("model", "demo")
    sliders = [
        {"name": "Stem 1", "gain": 1.0, "windows": [{"fmin": 100, "fmax": 1000}]},
        {"name": "Stem 2", "gain": 1.0, "windows": [{"fmin": 1000, "fmax": 4000}]},
        {"name": "Stem 3", "gain": 1.0, "windows": [{"fmin": 4000, "fmax": 8000}]},
        {"name": "Stem 4", "gain": 1.0, "windows": [{"fmin": 40, "fmax": 120}]},
    ]
    stems = [{"name": s["name"], "url": f"/api/audio/{sid}/output.wav"} for s in sliders]
    return _resp_json({"model": model, "sliders": sliders, "stems": stems})
# equalizer_app/views.py
import io, json, os, uuid, wave, math, struct, base64
from typing import List, Dict
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import numpy as np
# Importing the custom FFT algorithms
from .utils.fft import fft_radix2, ifft_radix2, stft_spectrogram

MEDIA_ROOT = getattr(settings, "MEDIA_ROOT", os.path.join(settings.BASE_DIR, "media"))
DATA_DIR = os.path.join(MEDIA_ROOT, "signals")
os.makedirs(DATA_DIR, exist_ok=True)

REG: Dict[str, dict] = {}


def _resp_json(data: dict):
    buf = json.dumps(data).encode("utf-8")
    return HttpResponse(buf, content_type="application/json")


def _write_wav(path, sr, x: np.ndarray):
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
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
    out_x = meta.get("output_x", meta["input_x"].copy())
    sr = meta["sr"]
    out_path = os.path.join(DATA_DIR, sid, "output.wav")
    _write_wav(out_path, sr, out_x)
    REG[sid]["output_x"] = out_x
    return out_path


@csrf_exempt
def upload_signal(request):
    if request.method != "POST": return HttpResponseBadRequest("POST only")
    f = request.FILES.get("signal")
    if not f: return HttpResponseBadRequest("No file 'signal'")

    sid = uuid.uuid4().hex[:12]
    sig_dir = os.path.join(DATA_DIR, sid)
    os.makedirs(sig_dir, exist_ok=True)

    orig_path = os.path.join(sig_dir, f.name)
    with open(orig_path, "wb") as fp:
        for chunk in f.chunks():
            fp.write(chunk)

    with open(orig_path, "rb") as fp:
        sr, x = _read_wav(fp)

    x_float = x.astype(np.float32)
    n = x_float.size
    n2 = _next_pow2(n)
    xz = np.zeros(n2, dtype=np.float32)
    xz[:n] = x_float
    input_X_complex = fft_radix2(xz)

    REG[sid] = {
        "file_name": f.name,
        "sr": int(sr),
        "input_x": x_float,
        "input_X_complex": input_X_complex,
        "output_x": x_float.copy(),
        "mode": "generic",
        "subbands": [],
        "custom_sliders": [],
        "scale": "linear",
        "show_spec": True,
    }
    _make_output_for_signal(sid)

    return _resp_json({
        "signal_id": sid, "file_name": f.name, "sr": int(sr),
        "n": int(x.size), "duration": float(x.size / sr),
    })


def summary(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    return _resp_json({
        "file_name": meta["file_name"], "sr": meta["sr"],
        "duration": float(meta["input_x"].size / meta["sr"]),
    })


def _compute_spectrum(x: np.ndarray, sr: int, scale: str):
    S = stft_spectrogram(x, sr)
    if S.size == 0: return [0.0], sr / 2.0
    mag = np.mean(S, axis=1)
    mag_max = mag.max()
    if mag_max > 1e-12: mag = mag / mag_max
    if scale == "audiogram": mag = np.power(mag, 0.7)
    return mag.tolist(), float(sr / 2.0)


def spectrum(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    scale = request.GET.get("scale", "linear")
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
    """
    Generates efficient Log-Scaled Spectrograms
    """
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    sr = meta["sr"]

    x_in = meta["input_x"]
    x_out = meta.get("output_x", x_in)

    # Get Linear STFT
    S_in = stft_spectrogram(x_in, sr)
    S_out = stft_spectrogram(x_out, sr)

    import PIL.Image as Image

    def process_and_encode(S):
        # 1. Log Magnitude for visibility (dB-like)
        # Using log1p to handle zeros gracefully
        S_log = np.log1p(S * 1000)

        # 2. Normalize [0..1]
        s_min, s_max = S_log.min(), S_log.max()
        if s_max - s_min > 1e-12:
            S_norm = (S_log - s_min) / (s_max - s_min)
        else:
            S_norm = np.zeros_like(S_log)

        # 3. Convert to [0..255] uint8
        img_data = (S_norm * 255).astype(np.uint8)

        # 4. Flip Y so 0Hz is at bottom
        img_data = img_data[::-1, :]

        # 5. Fast Vectorized Log-Frequency Warping
        h, w = img_data.shape
        if h > 1:
            # Create a mapping from dest_row -> src_row based on log scale
            # We want to stretch low freqs (bottom/high-index in flipped array)
            # and compress high freqs (top/low-index).

            # Normalized 0..1 coordinates
            y_lin = np.linspace(0, 1, h)

            # Apply power law for visual log-like effect (Audiogram style)
            # 0.25 power stretches the bottom significantly
            y_log = np.power(y_lin, 0.25)

            # Map to integer indices [0, h-1]
            src_indices = (y_log * (h - 1)).astype(int)

            # Clip to ensure bounds
            src_indices = np.clip(src_indices, 0, h - 1)

            # Apply warping using advanced indexing
            img_data = img_data[src_indices, :]

        im = Image.fromarray(img_data, mode="L")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return _resp_json({
        "in_png": process_and_encode(S_in),
        "out_png": process_and_encode(S_out)
    })


def custom_conf(request, sid):
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

    X = meta["input_X_complex"].copy()
    n = meta["input_x"].size
    n2 = X.size

    def apply_windows(windows, gain):
        for w in windows:
            fmin = max(0.0, float(w["fmin"]))
            fmax = max(0.0, float(w["fmax"]))
            if fmax < fmin: fmin, fmax = fmax, fmin
            kmin = int(fmin / (sr / n2))
            kmax = int(fmax / (sr / n2))
            kmin = max(0, min(kmin, n2 // 2 - 1))
            kmax = max(0, min(kmax, n2 // 2 - 1))
            if kmax < kmin: continue
            X[kmin:kmax + 1] *= gain
            if kmin > 0: X[-(kmax + 1):-(kmin)].__imul__(gain)

    if mode == "generic":
        subs = body.get("subbands", [])
        REG[sid]["subbands"] = subs
        for sb in subs: apply_windows([{"fmin": sb["fmin"], "fmax": sb["fmax"]}], float(sb["gain"]))
    else:
        sliders = body.get("sliders", [])
        REG[sid]["custom_sliders"] = sliders
        for s in sliders: apply_windows(s.get("windows", []), float(s.get("gain", 1.0)))

    xr = ifft_radix2(X).real[:n].astype(np.float32)
    REG[sid]["output_x"] = xr
    _make_output_for_signal(sid)
    return _resp_json({"ok": True})


@csrf_exempt
def save_scheme(request, sid):
    body = json.loads(request.body.decode("utf-8"))
    return _resp_json({"filename": f"scheme_{sid}.json", "data": body})


@csrf_exempt
def load_scheme(request, sid):
    body = json.loads(request.body.decode("utf-8"))
    REG[sid].update({"mode": body.get("mode", "generic"), "subbands": body.get("subbands", []),
                     "custom_sliders": body.get("sliders", [])})
    return _resp_json({"ok": True})


@csrf_exempt
def save_settings(request, sid):
    body = json.loads(request.body.decode("utf-8"))
    return _resp_json({"filename": f"settings_{sid}.json", "data": body})


@csrf_exempt
def load_settings(request, sid):
    body = json.loads(request.body.decode("utf-8"))
    REG[sid].update({"scale": body.get("scale", "linear"), "show_spec": True, "mode": body.get("mode", "generic"),
                     "subbands": body.get("subbands", []), "custom_sliders": body.get("sliders", [])})
    return _resp_json({"ok": True})


def audio_input(request, sid):
    meta = REG[sid]
    buf = io.BytesIO();
    _write_wav(buf, meta["sr"], meta["input_x"])
    return HttpResponse(buf.getvalue(), content_type="audio/wav")


def audio_output(request, sid):
    meta = REG[sid]
    buf = io.BytesIO();
    _write_wav(buf, meta["sr"], meta.get("output_x", meta["input_x"]))
    return HttpResponse(buf.getvalue(), content_type="audio/wav")


@csrf_exempt
def ai_run(request, sid):
    return _resp_json({"model": "demo", "sliders": [], "stems": []})
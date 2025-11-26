import io, json, os, uuid, wave, math, struct, base64
from typing import List, Dict
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import numpy as np
from .utils.fft import fft_radix2, ifft_radix2, stft_spectrogram
from .utils.fft_bridge import fft_cpp

MEDIA_ROOT = getattr(settings, "MEDIA_ROOT", os.path.join(settings.BASE_DIR, "media"))
DATA_DIR = os.path.join(MEDIA_ROOT, "signals")
CONFIG_DIR = os.path.join(settings.BASE_DIR, "equalizer_app", "configs")
os.makedirs(DATA_DIR, exist_ok=True)

REG: Dict[str, dict] = {}


def _resp_json(data: dict):
    buf = json.dumps(data).encode("utf-8")
    return HttpResponse(buf, content_type="application/json")


def _write_wav(path, sr, x: np.ndarray):
    x = x.astype(np.float32)
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
    if nchan > 1: x = x.reshape(-1, nchan).mean(axis=1)
    return sr, x


def _downsample_preview(x: np.ndarray, target=2000):
    if x.size <= target: return x.tolist()
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
        for chunk in f.chunks(): fp.write(chunk)

    with open(orig_path, "rb") as fp:
        sr, x = _read_wav(fp)

    x_float = x.astype(np.float32)
    n = x_float.size
    n2 = _next_pow2(n)
    xz = np.zeros(n2, dtype=np.float32)
    xz[:n] = x_float

    # Default initial FFT using Numpy
    input_X_complex = np.fft.fft(xz)

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
    return _resp_json(
        {"signal_id": sid, "file_name": f.name, "sr": int(sr), "n": int(x.size), "duration": float(x.size / sr)})


def summary(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    return _resp_json(
        {"file_name": meta["file_name"], "sr": meta["sr"], "duration": float(meta["input_x"].size / meta["sr"])})


def _compute_spectrum(x: np.ndarray, sr: int, scale: str, backend: str):
    VISUAL_POINTS = 2000

    # 1. Windowing (Fixes spectral leakage/spikiness)
    window = np.hanning(len(x))
    x_windowed = x * window

    if backend == 'cpp':
        n2 = _next_pow2(len(x))
        xz = np.zeros(n2, dtype=np.float32)
        xz[:len(x)] = x_windowed
        X = fft_cpp(xz)
        mag = np.abs(X[:n2 // 2])
    else:
        # Numpy fallback
        X = np.fft.fft(x_windowed)
        mag = np.abs(X[:len(X) // 2])

    # 2. Remove DC Offset (Index 0) - CRITICAL FIX
    # Large DC offsets squash the rest of the graph when normalized.
    if len(mag) > 0:
        mag[0] = 0

    # 3. Normalize
    max_val = mag.max()
    if max_val > 1e-12:
        mag /= max_val

    # 4. Visibility Scaling
    if scale == "audiogram":
        # dB Scale
        mag_db = 20 * np.log10(mag + 1e-9)
        min_db = -80.0
        mag = (mag_db - min_db) / (0 - min_db)
        mag = np.clip(mag, 0, 1)
    else:
        # "Linear" Mode Enhanced
        # Use Square Root compression to lift high frequencies/low amplitudes
        # so they are visible, while keeping the 0-1 range.
        mag = np.sqrt(mag)

    # 5. Smooth Downsampling (Binning)
    if len(mag) > VISUAL_POINTS:
        step = len(mag) // VISUAL_POINTS
        cutoff = step * VISUAL_POINTS
        # Average chunks instead of skipping
        mag = mag[:cutoff].reshape(-1, step).mean(axis=1)

    return mag.tolist(), float(sr / 2.0)


def spectrum(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    scale = request.GET.get("scale", "linear")
    backend = request.GET.get("backend", "numpy")
    x_data = meta.get("output_x", meta["input_x"])
    mags, fmax = _compute_spectrum(x_data, meta["sr"], scale, backend)
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

    import PIL.Image as Image
    def process_and_encode(S):
        # Remove DC from Spectrogram too
        if S.shape[0] > 0: S[0, :] = 0

        S_log = np.log1p(S * 1000)
        s_min, s_max = S_log.min(), S_log.max()
        S_norm = (S_log - s_min) / (s_max - s_min) if (s_max - s_min > 1e-12) else np.zeros_like(S_log)
        img_data = (S_norm * 255).astype(np.uint8)
        img_data = img_data[::-1, :]
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

    filename = ""
    if "animal" in mode:
        filename = "animal_sounds.json"
    elif "music" in mode:
        filename = "musical_instruments.json"
    elif "human" in mode:
        filename = "human_voices.json"

    if filename:
        json_path = os.path.join(CONFIG_DIR, filename)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                sliders = json.load(f)
    return _resp_json({"sliders": sliders})


@csrf_exempt
def equalize(request, sid):
    if request.method != "POST": return HttpResponseBadRequest("POST only")
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    body = json.loads(request.body.decode("utf-8"))
    mode = body.get("mode", "generic")
    sr = meta["sr"]

    xz = meta["input_x"]
    n = xz.size
    n2 = _next_pow2(n)

    padded_x = np.zeros(n2, dtype=np.float32)
    padded_x[:n] = xz

    try:
        X = fft_cpp(padded_x)
    except:
        X = np.fft.fft(padded_x)

    freqs = np.fft.fftfreq(n2, d=1.0 / sr)

    def apply_windows(windows, gain):
        for w in windows:
            fmin, fmax = float(w["fmin"]), float(w["fmax"])
            if fmax < fmin: fmin, fmax = fmax, fmin
            mask = (np.abs(freqs) >= fmin) & (np.abs(freqs) <= fmax)
            X[mask] *= gain

    if mode == "generic":
        subs = body.get("subbands", [])
        REG[sid]["subbands"] = subs
        for sb in subs: apply_windows([{"fmin": sb["fmin"], "fmax": sb["fmax"]}], float(sb["gain"]))
    else:
        sliders = body.get("sliders", [])
        REG[sid]["custom_sliders"] = sliders
        for s in sliders: apply_windows(s.get("windows", []), float(s.get("gain", 1.0)))

    xr = np.fft.ifft(X).real[:n].astype(np.float32)

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
    buf = io.BytesIO()
    _write_wav(buf, meta["sr"], meta["input_x"])
    return HttpResponse(buf.getvalue(), content_type="audio/wav")


def audio_output(request, sid):
    meta = REG[sid]
    buf = io.BytesIO()
    _write_wav(buf, meta["sr"], meta.get("output_x", meta["input_x"]))
    return HttpResponse(buf.getvalue(), content_type="audio/wav")


@csrf_exempt
def ai_run(request, sid):
    return _resp_json({"model": "demo", "sliders": [], "stems": []})
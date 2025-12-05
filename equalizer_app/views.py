import io, json, os, uuid, wave, base64
from typing import List, Dict
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import numpy as np

# IMPORT THE NEW C++ BRIDGE
from .utils import fft_bridge
from .ai_models.orchestrator import AIOrchestrator

MEDIA_ROOT = getattr(settings, "MEDIA_ROOT", os.path.join(settings.BASE_DIR, "media"))
DATA_DIR = os.path.join(MEDIA_ROOT, "signals")
CONFIG_DIR = os.path.join(settings.BASE_DIR, "equalizer_app", "configs")
os.makedirs(DATA_DIR, exist_ok=True)

REG: Dict[str, dict] = {}


def _resp_json(data: dict):
    buf = json.dumps(data).encode("utf-8")
    return HttpResponse(buf, content_type="application/json")


def _write_wav(fileobj, sr, x: np.ndarray):
    """
    Writes numpy array to WAV.
    fileobj can be a path (str) or a file-like object (io.BytesIO).
    """
    x = x.astype(np.float32)
    x = np.clip(x, -1.0, 1.0)

    # Handle both string path and file-like objects
    if isinstance(fileobj, str):
        wf = wave.open(fileobj, "wb")
    else:
        wf = wave.open(fileobj, "wb")

    try:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        frames = (x * 32767.0).astype(np.int16).tobytes()
        wf.writeframes(frames)
    finally:
        wf.close()


def _read_wav(fileobj):
    """
    Reads WAV file into float32 mono numpy array.
    """
    try:
        import soundfile as sf
        is_path = isinstance(fileobj, (str, os.PathLike))
        if is_path:
            y, sr = sf.read(fileobj, dtype='float32')
        else:
            if hasattr(fileobj, 'seek'): fileobj.seek(0)
            y, sr = sf.read(fileobj, dtype='float32')
        if y.ndim > 1: y = y.mean(axis=1)
        return int(sr), y.astype(np.float32)
    except:
        if hasattr(fileobj, 'seek'): fileobj.seek(0)
        with wave.open(fileobj, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            width = wf.getsampwidth()

            # Handle different bit depths manually if soundfile fails
            if width == 2:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif width == 1:
                data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                # Fallback for 32-bit float or other
                try:
                    data = np.frombuffer(raw, dtype=np.float32)
                except:
                    data = np.zeros(wf.getnframes(), dtype=np.float32)

            # Mix to mono if needed
            if wf.getnchannels() > 1:
                data = data.reshape(-1, wf.getnchannels()).mean(axis=1)

            return int(sr), data


def _next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p


def _pad_signal(x):
    """Pads signal to next power of 2 for efficient C++ FFT"""
    n = x.size
    n2 = _next_pow2(n)
    if n2 == n: return x
    padded = np.zeros(n2, dtype=np.float32)
    padded[:n] = x
    return padded


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

    REG[sid] = {
        "file_name": f.name,
        "sr": int(sr),
        "input_x": x,
        "output_x": x.copy(),
        "mode": "generic",
        "subbands": [],
        "custom_sliders": [],
        "scale": "linear",
        "stem_data": {},
        "ai_active": False
    }
    _make_output_for_signal(sid)
    return _resp_json(
        {"signal_id": sid, "file_name": f.name, "sr": int(sr), "n": int(x.size), "duration": float(x.size / sr)})


def summary(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    return _resp_json(
        {"file_name": meta["file_name"], "sr": meta["sr"], "duration": float(meta["input_x"].size / meta["sr"])})


def spectrum(request, sid):
    # PURE C++ IMPLEMENTATION
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    scale = request.GET.get("scale", "linear")

    x_data = meta.get("output_x", meta["input_x"])
    padded_x = _pad_signal(x_data)

    # Call C++ to get visualization magnitudes
    mags, fmax = fft_bridge.get_spectrum_data(padded_x, meta["sr"], scale)

    return _resp_json({"mags": mags, "fmax": fmax})


def wave_previews(request, sid):
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")

    def downsample(x, target=2000):
        if x.size <= target: return x.tolist()
        idx = np.linspace(0, x.size - 1, target).astype(np.int64)
        return x[idx].tolist()

    return _resp_json({
        "input": downsample(meta["input_x"]),
        "output": downsample(meta.get("output_x", meta["input_x"]))
    })


def spectrograms(request, sid):
    # PURE C++ IMPLEMENTATION
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")
    sr = meta["sr"]

    import PIL.Image as Image

    def generate_png(x_sig):
        # Call C++ Bridge
        _, _, S = fft_bridge.get_spectrogram_matrix(x_sig, sr)

        if S.size == 0: return ""

        # Log scaling for visualization
        S_log = np.log1p(S * 1000)
        s_min, s_max = S_log.min(), S_log.max()
        S_norm = (S_log - s_min) / (s_max - s_min) if (s_max - s_min > 1e-9) else np.zeros_like(S_log)

        # Flip Y-axis (low freq at bottom)
        img_data = (S_norm * 255).astype(np.uint8)[::-1, :]

        im = Image.fromarray(img_data, mode="L")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return _resp_json({
        "in_png": generate_png(meta["input_x"]),
        "out_png": generate_png(meta.get("output_x", meta["input_x"]))
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
    # PURE C++ IMPLEMENTATION
    if request.method != "POST": return HttpResponseBadRequest("POST only")
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")

    body = json.loads(request.body.decode("utf-8"))
    mode = body.get("mode", "generic")

    # 1. AI Mixing Mode (Summation)
    if mode == "ai_mix":
        gains = body.get("gains", {})
        stems = meta.get("stem_data", {})
        input_len = len(meta["input_x"])
        mixed_signal = np.zeros(input_len, dtype=np.float32)

        for name, arr in stems.items():
            g = float(gains.get(name, 0.0))
            if g > 0:
                l = min(len(mixed_signal), len(arr))
                mixed_signal[:l] += arr[:l] * g

        REG[sid]["output_x"] = mixed_signal
        _make_output_for_signal(sid)
        return _resp_json({"ok": True})

    # 2. Spectral EQ Mode (C++)
    # Collect all bands to apply
    bands_to_apply = []

    if mode == "generic":
        subs = body.get("subbands", [])
        REG[sid]["subbands"] = subs
        # Generic subbands: {fmin, fmax, gain}
        for s in subs:
            bands_to_apply.append({"fmin": s["fmin"], "fmax": s["fmax"], "gain": s["gain"]})
    else:
        sliders = body.get("sliders", [])
        REG[sid]["custom_sliders"] = sliders
        # Custom sliders: list of windows
        for s in sliders:
            gain = float(s.get("gain", 1.0))
            for w in s.get("windows", []):
                bands_to_apply.append({"fmin": w["fmin"], "fmax": w["fmax"], "gain": gain})

    # Prepare signal (pad to power of 2 for C++ FFT)
    x_input = meta["input_x"]
    x_padded = _pad_signal(x_input)

    # EXECUTE C++ FILTER
    if len(bands_to_apply) > 0:
        x_out_padded = fft_bridge.process_equalizer(x_padded, meta["sr"], bands_to_apply)
    else:
        x_out_padded = x_padded.copy()

    # Crop padding back to original length
    REG[sid]["output_x"] = x_out_padded[:x_input.size]

    _make_output_for_signal(sid)
    return _resp_json({"ok": True})


@csrf_exempt
def run_ai(request, sid):
    # This invokes the orchestrator (external process manager)
    if request.method != "POST": return HttpResponseBadRequest("POST only")
    meta = REG.get(sid)
    if not meta: return HttpResponseBadRequest("Invalid id")

    body = json.loads(request.body.decode("utf-8"))
    mode = body.get("mode", "music")

    orchestrator = AIOrchestrator()
    input_path = os.path.join(DATA_DIR, sid, meta["file_name"])

    # Target Sample Rate (project SR)
    target_sr = meta["sr"]

    try:
        # === MUSIC MODE ===
        if mode == "music":
            output_dir = os.path.join(DATA_DIR, sid, "stems")
            result = orchestrator.separate_music(input_path, output_dir)

            meta["stem_data"] = {}
            stem_names = []

            stems_map = result.get("stems", {})
            for name, path in stems_map.items():
                if os.path.exists(path):
                    file_sr, x = _read_wav(path)

                    # Resample logic (naive linear interp if needed, strictly in Python for now as it's not FFT)
                    # For full C++ purity, resampling could be moved to C++ too, but it's time-domain.
                    if file_sr != target_sr:
                        # Simple resampling
                        duration = x.size / file_sr
                        target_len = int(duration * target_sr)
                        x = np.interp(
                            np.linspace(0, x.size - 1, target_len),
                            np.arange(x.size),
                            x
                        ).astype(np.float32)

                    meta["stem_data"][name] = x
                    stem_names.append(name)

            meta["ai_active"] = True
            return _resp_json({"status": "ok", "stems": stem_names})

        # === HUMAN MODE (Speaker Separation) ===
        elif mode == "human":
            output_dir = os.path.join(settings.BASE_DIR, "equalizer_app", "ai_models", "human", "output")

            # Cleanup old files
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path): os.unlink(file_path)
                    except:
                        pass
            else:
                os.makedirs(output_dir, exist_ok=True)

            result = orchestrator.separate_human_voices(input_path, output_dir)

            meta["stem_data"] = {}
            stem_names = []
            target_speakers_map = {1: "speaker 1", 2: "speaker 2", 3: "speaker 3", 4: "speaker 4"}

            detected_files = {}
            if "speakers" in result:
                for spk in result["speakers"]:
                    detected_files[spk["id"]] = spk["filename"]

            input_len = len(meta["input_x"])

            for i in range(1, 5):
                key = target_speakers_map[i]
                stem_names.append(key)

                if i in detected_files:
                    fname = detected_files[i]
                    fpath = os.path.join(output_dir, fname)

                    if os.path.exists(fpath):
                        file_sr, x = _read_wav(fpath)
                        # Resample
                        if file_sr != target_sr:
                            duration = x.size / file_sr
                            target_len = int(duration * target_sr)
                            x = np.interp(
                                np.linspace(0, x.size - 1, target_len),
                                np.arange(x.size),
                                x
                            ).astype(np.float32)

                        meta["stem_data"][key] = x
                    else:
                        meta["stem_data"][key] = np.zeros(input_len, dtype=np.float32)
                else:
                    meta["stem_data"][key] = np.zeros(input_len, dtype=np.float32)

            meta["ai_active"] = True
            return _resp_json({"status": "ok", "stems": stem_names})

    except Exception as e:
        print(f"AI Separation Error: {e}")
        return _resp_json({"status": "error", "message": str(e)})

    return _resp_json({"status": "error", "message": f"Mode '{mode}' not supported for AI separation"})


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
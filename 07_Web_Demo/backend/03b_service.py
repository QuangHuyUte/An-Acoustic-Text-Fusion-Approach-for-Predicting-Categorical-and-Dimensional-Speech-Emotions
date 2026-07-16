import argparse
import base64
import importlib.util
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
INFER_SCRIPT = HERE / "03b_infer.py"
MAX_BODY_BYTES = int(os.getenv("MODEL_03B_SERVICE_MAX_BODY_BYTES", str(40 * 1024 * 1024)))


def load_infer_module():
    spec = importlib.util.spec_from_file_location("speech_demo_03b_infer", INFER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import inference script: {INFER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer03b = load_infer_module()
started_at = time.time()
prewarm_lock = threading.Lock()
prewarm_state = {
    "done": False,
    "running": False,
    "error": None,
    "startedAt": None,
    "finishedAt": None,
    "seconds": None,
}


def json_response(handler, status_code, payload):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def read_json(handler):
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    if content_length > MAX_BODY_BYTES:
        raise ValueError(f"Request body too large. Limit is {MAX_BODY_BYTES // 1024 // 1024} MB.")
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def prewarm_models():
    with prewarm_lock:
        if prewarm_state["done"]:
            return dict(prewarm_state)
        if prewarm_state["running"]:
            return dict(prewarm_state)
        prewarm_state.update({
            "running": True,
            "error": None,
            "startedAt": time.time(),
            "finishedAt": None,
            "seconds": None,
        })

    t0 = time.time()
    try:
        registry = infer03b.load_live_model_registry()
        default_profile = (
            os.getenv("MODEL_PROFILE_ID")
            or registry.get("defaultProfileId")
            or getattr(infer03b, "DEFAULT_PROFILE_ID", "03b_frozen_text_5fold")
        )
        loaded_profile = infer03b.prewarm_profile(default_profile)

        if os.getenv("MODEL_03B_PREWARM_ASR", "1") == "1":
            infer03b.load_asr_model()

        local_e2v = getattr(infer03b, "DEFAULT_EMOTION2VEC_MODEL", None)
        if os.getenv("MODEL_03B_PREWARM_E2V", "1") == "1" and local_e2v is not None:
            model_dir = Path(local_e2v)
            if (model_dir / "emotion2vec_base.pt").exists():
                infer03b.load_funasr_emotion2vec(str(model_dir))

        with prewarm_lock:
            prewarm_state.update({
                "done": True,
                "running": False,
                "error": None,
                "finishedAt": time.time(),
                "seconds": round(time.time() - t0, 3),
                "profile": loaded_profile,
            })
        return dict(prewarm_state)
    except Exception as exc:
        with prewarm_lock:
            prewarm_state.update({
                "done": False,
                "running": False,
                "error": f"{type(exc).__name__}: {exc}",
                "finishedAt": time.time(),
                "seconds": round(time.time() - t0, 3),
            })
        return dict(prewarm_state)


def prewarm_in_background():
    thread = threading.Thread(target=prewarm_models, name="model-03b-prewarm", daemon=True)
    thread.start()
    return thread


class ModelServiceHandler(BaseHTTPRequestHandler):
    server_version = "SpeechDemo03BService/1.0"

    def log_message(self, fmt, *args):
        if os.getenv("MODEL_03B_SERVICE_LOG", "1") == "1":
            super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            json_response(self, 200, {
                "ok": True,
                "service": "03B persistent Python model service",
                "uptimeSeconds": round(time.time() - started_at, 3),
                "device": str(infer03b.DEVICE),
                "prewarm": dict(prewarm_state),
            })
            return
        json_response(self, 404, {"ok": False, "error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/prewarm":
                state = prewarm_models()
                json_response(self, 200 if state.get("error") is None else 500, {
                    "ok": state.get("error") is None,
                    "prewarm": state,
                })
                return

            if parsed.path == "/analyze":
                payload = read_json(self)
                audio_b64 = payload.get("audioWavBase64") or payload.get("audio") or ""
                if not audio_b64:
                    json_response(self, 400, {"ok": False, "error": "Missing audioWavBase64."})
                    return

                tmp_dir = Path(tempfile.gettempdir()) / "speech-demo-03b-service"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                stem = str(payload.get("sessionId") or f"speech-{int(time.time() * 1000)}")
                stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:80]
                audio_path = tmp_dir / f"{stem}.wav"
                audio_path.write_bytes(base64.b64decode(audio_b64))

                t0 = time.time()
                try:
                    result = infer03b.infer(
                        audio_path,
                        payload.get("transcript") or "",
                        auto_transcribe=bool(payload.get("autoTranscribe", True)),
                        model_profile_id=payload.get("modelProfileId") or getattr(infer03b, "DEFAULT_PROFILE_ID", "03b_frozen_text_5fold"),
                    )
                finally:
                    try:
                        audio_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                result["service"] = {
                    "mode": "persistent-python-service",
                    "seconds": round(time.time() - t0, 3),
                    "prewarm": dict(prewarm_state),
                }
                json_response(self, 200 if result.get("ok", False) else 500, result)
                return

            json_response(self, 404, {"ok": False, "error": "Not found"})
        except Exception as exc:
            json_response(self, 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("MODEL_03B_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MODEL_03B_SERVICE_PORT", "8765")))
    parser.add_argument("--prewarm", action="store_true")
    args = parser.parse_args()

    if args.prewarm or os.getenv("MODEL_03B_PREWARM_ON_START", "0") == "1":
        prewarm_in_background()

    server = ThreadingHTTPServer((args.host, args.port), ModelServiceHandler)
    print(json.dumps({
        "ok": True,
        "service": "03B persistent Python model service",
        "host": args.host,
        "port": args.port,
        "device": str(infer03b.DEVICE),
    }, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

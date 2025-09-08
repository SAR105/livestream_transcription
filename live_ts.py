# -------------------- SSE ASR (sequential, auto-sync via EXTINF) --------------------
# - Sequential per-segment processing (no concurrency)
# - Auto pacing: wait ~EXTINF minus compute-time before publishing
# - Optional translations via deep-translator
# - Handles fMP4 (.m4s + #EXT-X-MAP) and .ts
#
# Tiny knobs (kept minimal; no strategy to change):
#   M3U8_URL, MODEL_SIZE, DEVICE, COMPUTE_TYPE, SEG_POLL_SEC, EMIT_MIN_CHARS
#   TRANSCRIPT_TXT, FFMPEG_TIMEOUT
#   TRANSLATE_ENABLED, TARGET_LANGS="de,es,fr"
#   EXTINF_FALLBACK_SEC=6      # used only if playlist lacks #EXTINF
#   EXTINF_MULT=1.0            # gentle nudge if you need it; leave 1.0 for tight sync
#   EXTINF_DELAY_MIN=0, EXTINF_DELAY_MAX=12
#   BACKLOG_FASTPATH=1         # if >0, skip delay when backlog exists to catch up

import asyncio, subprocess, json, time, os, tempfile
from typing import Dict, Set, Optional, List, Tuple
from urllib.parse import urljoin

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import aiohttp

# Optional translations
_TRANSLATE_AVAILABLE = True
try:
    from deep_translator import GoogleTranslator
except Exception as _e:
    _TRANSLATE_AVAILABLE = False
    print(f"[translate] deep-translator not available ({_e}); translations disabled.")

# ---------------------- CONFIG ----------------------
M3U8_URL         = os.environ.get("M3U8_URL", "http://localhost:8888/streammain/stream.m3u8")
MODEL_SIZE       = os.environ.get("MODEL_SIZE", "small")
DEVICE           = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE     = os.environ.get("COMPUTE_TYPE", "int8")
LANGUAGE         = "en"
SAMPLE_RATE      = 16000
SEG_POLL_SEC     = float(os.environ.get("SEG_POLL_SEC", "2"))
TRANSCRIPT_TXT   = os.environ.get("TRANSCRIPT_TXT", "transcript.txt")
EMIT_MIN_CHARS   = int(os.environ.get("EMIT_MIN_CHARS", "6"))
FFMPEG_TIMEOUT   = float(os.environ.get("FFMPEG_TIMEOUT", "25"))

# Translation config
TRANSLATE_ENABLED = os.environ.get("TRANSLATE_ENABLED", "1") not in ("0", "false", "False", "")
TARGET_LANGS = [
    lang.strip().lower()
    for lang in os.environ.get("TARGET_LANGS", "de,es,fr").split(",")
    if lang.strip()
]

# Auto-delay tuning (kept minimal)
EXTINF_FALLBACK_SEC = float(os.environ.get("EXTINF_FALLBACK_SEC", "6"))
EXTINF_MULT         = float(os.environ.get("EXTINF_MULT", "1.0"))
EXTINF_DELAY_MIN    = float(os.environ.get("EXTINF_DELAY_MIN", "0"))
EXTINF_DELAY_MAX    = float(os.environ.get("EXTINF_DELAY_MAX", "12"))
BACKLOG_FASTPATH    = os.environ.get("BACKLOG_FASTPATH", "1") not in ("0", "false", "False", "")

# ----------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------- Fan-out broadcast ----------
class Fanout:
    def __init__(self, max_queue: int = 200):
        self._subs: Dict[int, asyncio.Queue] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._max_queue = max_queue

    async def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=self._max_queue)
        async with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._subs[sid] = q
        q._sid = sid  # type: ignore[attr-defined]
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        sid = getattr(q, "_sid", None)
        async with self._lock:
            if sid in self._subs:
                del self._subs[sid]

    async def publish(self, msg: dict):
        async with self._lock:
            subs = list(self._subs.values())
        for q in subs:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

    async def subscriber_count(self) -> int:
        async with self._lock:
            return len(self._subs)

bcast = Fanout()

# ---------- ASR model ----------
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# ---------- HTTP helpers ----------
async def fetch_text(session: aiohttp.ClientSession, url: str, timeout_sec: float = 5.0) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
        r.raise_for_status()
        return await r.text()

async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout_sec: float = 10.0) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
        r.raise_for_status()
        return await r.read()

# ---------- media/ASR helpers (blocking -> run via to_thread) ----------
def run_ffmpeg_to_pcm_blocking(input_path_or_url: str) -> np.ndarray:
    cmd = [
        "ffmpeg", "-hide_banner",
        "-loglevel", "error",
        "-i", input_path_or_url,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate(timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"ffmpeg decode timeout ({FFMPEG_TIMEOUT}s) for {input_path_or_url}")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: rc={proc.returncode}, err={err.decode('utf-8','ignore')}")
    if not out:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(out, dtype=np.int16)

def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    if pcm.dtype != np.int16:
        raise ValueError("expected int16 pcm")
    return pcm.astype(np.float32) / 32768.0

def transcribe_audio_block_blocking(audio_f32: np.ndarray) -> str:
    segments, _ = model.transcribe(
        audio_f32,
        language=LANGUAGE,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
        condition_on_previous_text=False,
        beam_size=5,
        word_timestamps=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text

# ---------- Translation helper ----------
async def translate_text_multi(source_lang: str, text: str, targets: List[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    if not text or not targets or not _TRANSLATE_AVAILABLE:
        return results

    async def _do_one(tgt: str) -> Optional[str]:
        try:
            return await asyncio.to_thread(GoogleTranslator(source=source_lang, target=tgt).translate, text)
        except Exception as e:
            print(f"[translate] {source_lang}->{tgt} error: {e}")
            return None

    tasks = [asyncio.create_task(_do_one(t)) for t in targets]
    outs = await asyncio.gather(*tasks, return_exceptions=False)
    for lang, val in zip(targets, outs):
        if isinstance(val, str) and val:
            results[lang] = val
    return results

# ---------- Seq / file guards ----------
SEQ_LOCK = asyncio.Lock()
FILE_LOCK = asyncio.Lock()
last_emitted_seq = 0  # guarded by SEQ_LOCK

def _append_to_file_sync(path: str, text: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")

async def append_to_file(path: str, text: str):
    async with FILE_LOCK:
        await asyncio.to_thread(_append_to_file_sync, path, text)

# ---------- Delay helpers ----------
def _cap(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

async def apply_extinf_delay(extinf: Optional[float], compute_elapsed: float, backlog_remaining: int):
    """
    Delay AFTER transcript is ready but BEFORE publishing, using (#EXTINF * MULT) - compute_time.
    If backlog exists and BACKLOG_FASTPATH=1, skip delay to catch up.
    """
    if BACKLOG_FASTPATH and backlog_remaining > 0:
        return  # catch up first

    base = extinf if (extinf is not None and extinf > 0) else EXTINF_FALLBACK_SEC
    desired = max(0.0, (base * EXTINF_MULT) - compute_elapsed)
    delay = _cap(desired, EXTINF_DELAY_MIN, EXTINF_DELAY_MAX)
    if delay > 0:
        await asyncio.sleep(delay)

# ---------- Per-segment processing (SEQUENTIAL) ----------
async def process_segment(seg_url: str, init_bytes: Optional[bytes], extinf: Optional[float], backlog_remaining: int):
    t0 = time.time()
    try:
        # Decode to PCM (blocking → default thread)
        if seg_url.endswith(".m4s") and init_bytes is not None:
            def _rebuild_and_decode():
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
                    tf.write(init_bytes)
                    import urllib.request
                    with urllib.request.urlopen(seg_url, timeout=10) as r:
                        seg_bytes = r.read()
                    tf.write(seg_bytes)
                    tf.flush()
                    return run_ffmpeg_to_pcm_blocking(tf.name)
            pcm16 = await asyncio.to_thread(_rebuild_and_decode)
        else:
            pcm16 = await asyncio.to_thread(run_ffmpeg_to_pcm_blocking, seg_url)

        if pcm16.size == 0:
            print(f"[segment] empty audio decoded, skipping: {seg_url}")
            return

        audio_f32 = pcm16_to_float32(pcm16)
        text_out = await asyncio.to_thread(transcribe_audio_block_blocking, audio_f32)
        if len(text_out) < EMIT_MIN_CHARS:
            return

        t_before_delay = time.time()
        compute_elapsed = t_before_delay - t0

        # Allocate sequence
        async with SEQ_LOCK:
            global last_emitted_seq
            last_emitted_seq += 1
            seq = last_emitted_seq

        # Build transcript payload (EN + optional translations)
        transcripts: Dict[str, str] = {"en": text_out}
        if TRANSLATE_ENABLED and TARGET_LANGS:
            try:
                tr_map = await translate_text_multi("en", text_out, [t for t in TARGET_LANGS if t != "en"])
                if tr_map:
                    transcripts.update(tr_map)
            except Exception as e:
                print(f"[translate] batch error: {e}")

        msg = {"seq": seq, "transcript": transcripts}

        # --- Auto delay here (EXTINF minus compute) ---
        await apply_extinf_delay(extinf, compute_elapsed, backlog_remaining)

        # Publish SSE + append English to file
        await bcast.publish(msg)
        await append_to_file(TRANSCRIPT_TXT, text_out)

        print(f"[segment] emitted seq={seq} (extinf={extinf}, compute={compute_elapsed:.2f}s, backlog={backlog_remaining}) -> {seg_url}")

    except Exception as e:
        print(f"[segment] error processing {seg_url}: {e}")

# ---------- Playlist watcher (SEQUENTIAL) ----------
_playlist_task: Optional[asyncio.Task] = None
_seen: Set[str] = set()
_init_bytes: Optional[bytes] = None
_watch_stop_evt = asyncio.Event()

def _parse_playlist(lines: List[str], base_url: str) -> Tuple[List[Tuple[str, Optional[float]]], Optional[str]]:
    """
    Returns:
      segs: list of (absolute_url, extinf_duration)
      init_uri: absolute init url if EXT-X-MAP seen; else None
    """
    segs: List[Tuple[str, Optional[float]]] = []
    init_uri_abs: Optional[str] = None
    pending_extinf: Optional[float] = None

    for ln in lines:
        if not ln:
            continue
        if ln.startswith("#EXT-X-MAP:"):
            try:
                part = next(p for p in ln.split(":", 1)[1].split(",") if p.strip().startswith("URI="))
                uri_val = part.split("=", 1)[1].strip().strip('"')
                init_uri_abs = urljoin(base_url, uri_val)
            except Exception as e:
                print(f"[playlist] failed to parse EXT-X-MAP: {e}")
        elif ln.startswith("#EXTINF:"):
            try:
                dur_str = ln.split(":", 1)[1].split(",")[0].strip()
                pending_extinf = float(dur_str)
            except Exception:
                pending_extinf = None
        elif ln.startswith("#"):
            continue
        else:
            seg_url = urljoin(base_url, ln.strip())
            segs.append((seg_url, pending_extinf))
            pending_extinf = None

    return segs, init_uri_abs

async def playlist_watcher():
    global _init_bytes
    _watch_stop_evt.clear()

    async with aiohttp.ClientSession() as session:
        while not _watch_stop_evt.is_set():
            try:
                text = await fetch_text(session, M3U8_URL, timeout_sec=5.0)
            except Exception as e:
                print(f"[playlist] fetch error: {e}")
                await asyncio.sleep(SEG_POLL_SEC)
                continue

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            segs, init_abs = _parse_playlist(lines, M3U8_URL)

            # Cache init for fMP4 once
            if init_abs and _init_bytes is None:
                try:
                    _init_bytes = await fetch_bytes(session, init_abs, timeout_sec=10.0)
                    print(f"[playlist] cached init bytes from {init_abs} ({len(_init_bytes)} bytes)")
                except Exception as e:
                    print(f"[playlist] init fetch error: {e}")

            # Discover new segments (preserve playlist order)
            new_items: List[Tuple[str, Optional[float]]] = []
            for url, dur in segs:
                if url not in _seen:
                    _seen.add(url)
                    new_items.append((url, dur))

            # prune memory
            if len(_seen) > 2000:
                for u in list(_seen)[:len(_seen)-1500]:
                    _seen.remove(u)

            # --- SEQUENTIAL: process one-by-one ---
            total = len(new_items)
            for i, (seg_url, extinf) in enumerate(new_items):
                backlog_remaining = (total - i - 1)
                await process_segment(seg_url, _init_bytes, extinf, backlog_remaining)

            await asyncio.sleep(SEG_POLL_SEC)

    print("[watcher] exited")

async def ensure_watcher_running():
    global _playlist_task
    if _playlist_task is None or _playlist_task.done():
        print("[watcher] starting (lazy on first subscriber)")
        _playlist_task = asyncio.create_task(playlist_watcher())

async def maybe_stop_watcher():
    if await bcast.subscriber_count() == 0:
        print("[watcher] stopping (no subscribers)")
        _watch_stop_evt.set()
        if _playlist_task:
            try:
                await asyncio.wait_for(_playlist_task, timeout=2.0)
            except Exception:
                pass

# ---------- SSE ----------
@app.get("/sse")
async def sse(request: Request):
    await ensure_watcher_running()
    q = await bcast.subscribe()

    async def gen():
        try:
            yield ": ok\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        finally:
            await bcast.unsubscribe(q)
            await maybe_stop_watcher()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ---------- Health ----------
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

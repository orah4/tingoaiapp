# app.py
from typing import Optional

from fastapi import (
    FastAPI, Form, Request, Response, UploadFile, File,
    APIRouter, Depends, Header, Query, HTTPException, status
)
from fastapi.responses import (
    JSONResponse, HTMLResponse, RedirectResponse,
    PlainTextResponse, StreamingResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI

import requests
import os, json, uuid, datetime, logging, traceback, mimetypes, shutil, time
import hmac, hashlib, base64


# ---------- Boot ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to .env or Render env")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN missing. Add it to .env or Render env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Harmonix")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use persistent disk if ALBUM_DIR is set (Render disk), else local folder.
ALBUM_DIR = os.getenv("ALBUM_DIR") or os.path.join(BASE_DIR, "album")

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "static")
ALBUM_JSON    = os.path.join(ALBUM_DIR, "album.json")
JOBS_JSON     = os.path.join(ALBUM_DIR, "jobs.json")

# Control whether to expose /album publicly (default: yes for UI)
PUBLIC_ALBUM = os.getenv("PUBLIC_ALBUM", "1") == "1"

# Streaming secret for signed links
STREAM_SECRET = os.getenv("PUBLIC_STREAM_SECRET", "")
if not STREAM_SECRET:
    logging.getLogger("harmonix").warning(
        "PUBLIC_STREAM_SECRET is not set — signed streaming will not work until you set it."
    )

# -------- Public API keys (env OR secret file) ----------
def _load_api_keys() -> set[str]:
    """
    Reads PUBLIC_API_KEYS from either:
      - env var PUBLIC_API_KEYS="k1,k2"
      - Render Secret File /etc/secrets/PUBLIC_API_KEYS (one or many keys)
    Accepts comma- or newline-separated lists.
    """
    keys_str = (os.getenv("PUBLIC_API_KEYS") or "").strip()
    if not keys_str:
        try:
            with open("/etc/secrets/PUBLIC_API_KEYS", "r", encoding="utf-8") as fh:
                keys_str = fh.read().strip()
        except Exception:
            keys_str = ""
    parts: list[str] = []
    for line in keys_str.replace("\r", "\n").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    return {p for p in parts if p}

API_KEYS = _load_api_keys()

def require_api_key(
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None),
):
    token = (x_api_key or api_key or "").strip()
    if not API_KEYS:
        raise HTTPException(status_code=503, detail="API not configured (no PUBLIC_API_KEYS).")
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return token


# ---------- FS bootstrap ----------
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(ALBUM_DIR, exist_ok=True)
for f, init in [(ALBUM_JSON, []), (JOBS_JSON, {})]:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(init, fh, indent=2)

# Static mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if PUBLIC_ALBUM:
    app.mount("/album", StaticFiles(directory=ALBUM_DIR), name="album")
else:
    logging.getLogger("harmonix").info("PUBLIC_ALBUM=0 → Not mounting /album publicly.")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harmonix")

# ---------- External model config ----------
BARK_VERSION = "b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787"
REPLICATE_HEADERS = {
    "Authorization": f"Token {REPLICATE_API_TOKEN}",
    "Content-Type": "application/json",
}
ALLOWED_UPLOAD_CTYPES = {
    "audio/wav": "wav", "audio/x-wav": "wav",
    "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/mp4": "m4a", "audio/aac": "m4a",
}

# ---------- Helpers ----------
BANNED_STYLE_WORDS = [" by ", " in the style of ", " sound like ", " like ", " as ", " ft. ", " featuring "]

def sanitize_prompt(p: str) -> str:
    s = " " + (p or "").strip().lower() + " "
    for w in BANNED_STYLE_WORDS:
        if w in s:
            s = s.split(w)[0] + " "
            break
    s = s.strip()
    return (f"{s}. Afrobeats vibe inspired by modern Lagos club sound, "
            "original melody & lyrics, no imitation of any specific artist.")

def album_read() -> list:
    with open(ALBUM_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def album_write(data: list) -> None:
    with open(ALBUM_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def append_to_album(entry: dict) -> None:
    data = album_read()
    data.append(entry)
    album_write(data)

def jobs_read() -> dict:
    with open(JOBS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def jobs_write(j: dict) -> None:
    with open(JOBS_JSON, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2)

def list_album_files() -> list:
    items = []
    try:
        for name in sorted(os.listdir(ALBUM_DIR)):
            path = os.path.join(ALBUM_DIR, name)
            if os.path.isfile(path):
                st = os.stat(path)
                items.append({
                    "name": name,
                    "url": f"/album/{name}",
                    "size_bytes": st.st_size,
                    "size_mb": round(st.st_size / (1024*1024), 3),
                    "modified": datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
                })
    except Exception as e:
        logger.error("List storage error: %s", e)
    return items

# ---------- Lyrics ----------
def generate_lyrics(prompt: str) -> str:
    clean = sanitize_prompt(prompt)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ("You are a concise lyricist for brand-new songs. "
                             "Write catchy Afrobeats lyrics under 60 words. "
                             "Do NOT imitate or name specific living artists. "
                             "Return ONLY the lyrics.")},
                {"role": "user", "content": clean}
            ],
            temperature=0.9,
            max_tokens=120,
        )
        lyrics = resp.choices[0].message.content.strip()
        if not lyrics:
            raise RuntimeError("Empty lyrics from OpenAI.")
        return lyrics
    except Exception as e:
        logger.error("OpenAI error: %s\n%s", e, traceback.format_exc())
        return "Rhythms rise under Lagos skies, heartbeats dancing in the night; we sing our dreams till morning light."

# ---------- Replicate create/poll ----------
def replicate_create_prediction(prompt_text: str) -> str:
    create_url = "https://api.replicate.com/v1/predictions"
    payload = {
        "version": BARK_VERSION,
        "input": {
            "prompt": prompt_text,
            "text_temp": 0.7,
            "waveform_temp": 0.7,
            "output_full": False
        }
    }
    resp = requests.post(create_url, headers=REPLICATE_HEADERS, json=payload, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Replicate create failed: {resp.status_code} {resp.text}")
    pred = resp.json()
    pred_id = pred.get("id")
    if not pred_id:
        raise RuntimeError(f"Replicate create returned no id: {pred}")
    return pred_id

def replicate_get_prediction(pred_id: str) -> dict:
    url = f"https://api.replicate.com/v1/predictions/{pred_id}"
    try:
        g = requests.get(url, headers=REPLICATE_HEADERS, timeout=30)
        if g.status_code != 200:
            return {"status": "poll_error", "http": g.status_code, "text": g.text}
        return g.json()
    except requests.Timeout:
        return {"status": "poll_timeout"}

# ---------- Download helpers ----------
def infer_ext_from_headers_or_url(ctype: str, url: str) -> str:
    ctype = (ctype or "").lower()
    if ctype in ALLOWED_UPLOAD_CTYPES:
        return ALLOWED_UPLOAD_CTYPES[ctype]
    guessed = mimetypes.guess_extension(ctype)
    if guessed:
        return guessed.replace(".", "")
    last = url.split("?")[0].split("#")[0].split(".")[-1].lower()
    return last if last and len(last) <= 5 else "wav"

def download_to_album(url: str) -> str:
    r = requests.get(url, timeout=300)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Download failed: {r.status_code}")
    ext = infer_ext_from_headers_or_url(r.headers.get("Content-Type"), url)
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(ALBUM_DIR, filename)
    with open(path, "wb") as f:
        f.write(r.content)
    return path

def stream_import_to_album(url: str, max_mb: int = 64) -> str:
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Import GET failed: {r.status_code}")
        ext = infer_ext_from_headers_or_url(r.headers.get("Content-Type"), url)
        filename = f"{uuid.uuid4().hex}.{ext}"
        path = os.path.join(ALBUM_DIR, filename)
        max_bytes = max_mb * 1024 * 1024
        total = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    os.remove(path)
                    raise RuntimeError(f"File too large (> {max_mb} MB).")
                f.write(chunk)
    return path

# ---------- Basic pages & album listing ----------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat() + "Z", "album_dir": ALBUM_DIR}

@app.get("/api/album")
async def get_album():
    return list(reversed(album_read()))  # newest first

# ---- Storage viewers ----
@app.get("/api/storage")
async def api_storage():
    return {"album_dir": ALBUM_DIR, "files": list_album_files()}

@app.get("/storage", response_class=HTMLResponse)
async def storage_view():
    files = list_album_files()
    html_rows = []
    for f in files:
        html_rows.append(
            f"<tr><td><a href='{f['url']}'>{f['name']}</a></td>"
            f"<td style='text-align:right'>{f['size_mb']}</td>"
            f"<td>{f['modified']}</td></tr>"
        )
    html = f"""
    <html>
    <head>
      <title>Album Storage</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background:#111; color:#eee; }}
        table {{ border-collapse: collapse; width:100%; }}
        th,td {{ border:1px solid #333; padding:8px; }}
        a {{ color:#6cf; text-decoration:none; }}
      </style>
    </head>
    <body>
      <h2>Album Storage</h2>
      <p><b>Album dir:</b> {ALBUM_DIR}</p>
      <table>
        <thead><tr><th>File</th><th style='text-align:right'>Size (MB)</th><th>Modified (UTC)</th></tr></thead>
        <tbody>{''.join(html_rows) if html_rows else "<tr><td colspan=3>(empty)</td></tr>"}</tbody>
      </table>
      <p>Tip: Direct links above are served from <code>/album/…</code></p>
    </body>
    </html>
    """
    return HTMLResponse(html)

# ---------- Job helpers ----------
def start_generation_job(prompt: str, title: Optional[str] = None, duration: int = 0) -> dict:
    """Starts a Replicate job and persists it in jobs.json. Returns the job dict."""
    lyrics = generate_lyrics(prompt)
    logger.info("Lyrics OK: %s", lyrics[:80])

    pred_id = replicate_create_prediction(lyrics)
    job_id = uuid.uuid4().hex
    created_at = datetime.datetime.utcnow().isoformat() + "Z"

    jobs = jobs_read()
    job = {
        "id": job_id,
        "pred_id": pred_id,
        "status": "starting",
        "lyrics": lyrics,
        "title": title or sanitize_prompt(prompt)[:40],
        "created_at": created_at,
        "saved": False,
    }
    jobs[job_id] = job
    jobs_write(jobs)
    return job

# ---------- UI routes for generation ----------
@app.post("/generate-music")
async def generate_music(
    prompt: str = Form(...),
    title: str = Form(None),
    duration: int = Form(0),
):
    try:
        job = start_generation_job(prompt, title, duration)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"message": "Job created", "job": job}
        )
    except Exception as e:
        logger.error("Create job error: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "create_failed", "detail": str(e)})

@app.get("/job/{job_id}")
async def get_job(job_id: str):
    jobs = jobs_read()
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "not_found"})

    if job.get("status") == "succeeded" and job.get("saved") and job.get("track"):
        return {"job": job, "track": job["track"]}

    pred = replicate_get_prediction(job["pred_id"])
    status_s = pred.get("status", "unknown")
    if status_s in ("poll_timeout", "poll_error"):
        return {"job": job, "note": status_s}

    job["status"] = status_s
    job["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    if status_s == "succeeded" and not job.get("saved"):
        output = pred.get("output")
        url = None
        if isinstance(output, list) and output:
            url = output[0]
        elif isinstance(output, dict):
            url = output.get("audio_out")
        if not url:
            job["status"] = "failed"
            jobs[job_id] = job
            jobs_write(jobs)
            return JSONResponse(status_code=500, content={"error": "no_audio_url"})
        try:
            local_path = download_to_album(url)
            public_url = f"/album/{os.path.basename(local_path)}"
            track_entry = {
                "id": uuid.uuid4().hex,
                "title": job["title"],
                "artist": "",
                "lyrics": job["lyrics"],
                "file": os.path.basename(local_path),
                "url": public_url,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "source": "bark",
            }
            append_to_album(track_entry)
            job["track"] = track_entry
            job["saved"] = True
        except Exception as e:
            job["status"] = "failed"
            job["error"] = f"download_failed: {e}"

    jobs[job_id] = job
    jobs_write(jobs)

    if job["status"] == "succeeded" and job.get("track"):
        return {"job": job, "track": job["track"]}
    elif job["status"] in ("failed", "canceled"):
        return JSONResponse(status_code=500, content={"job": job, "error": job.get("error", job["status"])})
    else:
        return {"job": job}

# ---------- Album utilities (UI + API reuse) ----------
@app.post("/api/album/upload")
async def api_album_upload(
    title: str = Form(None),
    artist: str = Form(None),
    file: UploadFile = File(...),
):
    try:
        ctype = (file.content_type or "").lower()
        if not ctype.startswith("audio/"):
            return JSONResponse(status_code=400, content={"error": "invalid_type", "detail": "Please upload an audio file."})

        ext = ALLOWED_UPLOAD_CTYPES.get(ctype)
        if not ext:
            name_ext = os.path.splitext(file.filename or "")[1].lower().replace(".", "")
            if name_ext in ("mp3", "wav", "ogg", "webm", "m4a", "aac"):
                ext = name_ext
            else:
                ext = (mimetypes.guess_extension(ctype) or ".wav").replace(".", "")

        filename = f"{uuid.uuid4().hex}.{ext}"
        path = os.path.join(ALBUM_DIR, filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        public_url = f"/album/{filename}"
        entry = {
            "id": uuid.uuid4().hex,
            "title": title or (file.filename or "Untitled"),
            "artist": artist or "",
            "lyrics": "",
            "file": filename,
            "url": public_url,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source": "upload",
        }
        append_to_album(entry)
        return {"track": entry}
    except Exception as e:
        logger.error("Upload error: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "upload_failed", "detail": str(e)})

@app.post("/api/album/import_url")
async def api_album_import_url(request: Request):
    """Download a direct audio URL into the album."""
    try:
        body = await request.json()
        url = (body.get("url") or "").strip()
        title = body.get("title") or "Imported Track"
        artist = body.get("artist") or ""

        if not url:
            return JSONResponse(status_code=400, content={"error": "url_required"})

        local_path = stream_import_to_album(url, max_mb=64)
        public_url = f"/album/{os.path.basename(local_path)}"
        entry = {
            "id": uuid.uuid4().hex,
            "title": title,
            "artist": artist,
            "lyrics": "",
            "file": os.path.basename(local_path),
            "url": public_url,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source": "import_url",
            "origin_url": url,
        }
        append_to_album(entry)
        return {"track": entry}
    except Exception as e:
        logger.error("Import URL error: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "import_failed", "detail": str(e)})

@app.delete("/api/album/{track_id}")
async def api_album_delete(track_id: str):
    """Delete a track from album (and the local file if present)."""
    try:
        data = album_read()
        idx = next((i for i, t in enumerate(data) if t.get("id") == track_id), None)
        if idx is None:
            return JSONResponse(status_code=404, content={"error": "not_found"})
        track = data.pop(idx)
        fn = track.get("file")
        if fn:
            path = os.path.join(ALBUM_DIR, fn)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        album_write(data)
        return {"deleted": True}
    except Exception as e:
        logger.error("Delete error: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "delete_failed", "detail": str(e)})

# ---------- Signing helpers (for secure streaming) ----------
def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")

def _unpad_b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def sign_stream(filename: str, exp_epoch: int) -> str:
    """HMAC-SHA256 over '<filename>:<exp>' using STREAM_SECRET."""
    msg = f"{filename}:{exp_epoch}".encode()
    key = STREAM_SECRET.encode()
    sig = hmac.new(key, msg, hashlib.sha256).digest()
    return _b64url(sig)

def verify_stream_sig(filename: str, exp_epoch: int, sig: str) -> bool:
    if not STREAM_SECRET:
        return False
    try:
        msg = f"{filename}:{exp_epoch}".encode()
        key = STREAM_SECRET.encode()
        expected = hmac.new(key, msg, hashlib.sha256).digest()
        provided = _unpad_b64(sig)
        if time.time() > exp_epoch:
            return False
        return hmac.compare_digest(expected, provided)
    except Exception:
        return False

# ---------------- Public API v1 ----------------
# (We attach the API key requirement PER route so /stream can allow signed URLs without a key.)
api = APIRouter(prefix="/api/v1", tags=["Public API"])

@api.get("/health", dependencies=[Depends(require_api_key)])
def api_health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat() + "Z"}

@api.post("/generate", dependencies=[Depends(require_api_key)])
def api_generate(
    prompt: str = Form(...),
    title: Optional[str] = Form(None),
    duration: int = Form(0)
):
    job = start_generation_job(prompt, title, duration)
    return {"job": job, "poll_url": f"/api/v1/jobs/{job['id']}"}

@api.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def api_job(job_id: str):
    jobs = jobs_read()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="not_found")

    if job.get("status") == "succeeded" and job.get("saved") and job.get("track"):
        return {"job": job, "track": job["track"]}

    pred = replicate_get_prediction(job["pred_id"])
    status_s = pred.get("status", "unknown")
    if status_s in ("poll_timeout", "poll_error"):
        return {"job": job, "note": status_s}

    job["status"] = status_s
    job["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    if status_s == "succeeded" and not job.get("saved"):
        output = pred.get("output")
        url = None
        if isinstance(output, list) and output:
            url = output[0]
        elif isinstance(output, dict):
            url = output.get("audio_out")
        if not url:
            job["status"] = "failed"
            jobs[job_id] = job
            jobs_write(jobs)
            raise HTTPException(status_code=500, detail="no_audio_url")

        local_path = download_to_album(url)
        public_url = f"/album/{os.path.basename(local_path)}"
        track_entry = {
            "id": uuid.uuid4().hex,
            "title": job["title"],
            "artist": "",
            "lyrics": job["lyrics"],
            "file": os.path.basename(local_path),
            "url": public_url,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source": "bark",
        }
        append_to_album(track_entry)
        job["track"] = track_entry
        job["saved"] = True

    jobs[job_id] = job
    jobs_write(jobs)

    if job["status"] == "succeeded" and job.get("track"):
        return {"job": job, "track": job["track"]}
    elif job["status"] in ("failed", "canceled"):
        raise HTTPException(status_code=500, detail=job.get("error", job["status"]))
    else:
        return {"job": job}

@api.get("/tracks", dependencies=[Depends(require_api_key)])
def api_tracks(request: Request, ttl: int = Query(3600, ge=60, le=86400), signed: bool = Query(False)):
    """List tracks (newest first). If signed=1, include short-lived secure_url for each."""
    data = list(reversed(album_read()))
    if not signed:
        return data

    base = str(request.base_url).rstrip("/")
    now = int(time.time())
    out = []
    for t in data:
        rel = t.get("url") or ""
        if not rel:
            continue
        filename = rel.rsplit("/", 1)[-1]
        item = dict(t)
        if STREAM_SECRET:
            exp = now + ttl
            sig = sign_stream(filename, exp)
            item["secure_url"] = f"{base}/api/v1/stream/{filename}?exp={exp}&sig={sig}"
        else:
            item["secure_url"] = f"{base}/api/v1/stream/{filename}?api_key=YOUR_KEY_HERE"
        out.append(item)
    return out

@api.get("/tracks/{track_id}", dependencies=[Depends(require_api_key)])
def api_track(track_id: str):
    data = album_read()
    track = next((t for t in data if t.get("id") == track_id), None)
    if not track:
        raise HTTPException(status_code=404, detail="not_found")
    return track

@api.get("/tracks/{track_id}/download", dependencies=[Depends(require_api_key)])
def api_track_download(track_id: str):
    data = album_read()
    track = next((t for t in data if t.get("id") == track_id), None)
    if not track:
        raise HTTPException(status_code=404, detail="not_found")
    return RedirectResponse(url=track["url"])

@api.post("/upload", dependencies=[Depends(require_api_key)])
async def api_upload_public(
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    return await api_album_upload(title=title, artist=artist, file=file)

@api.post("/import", dependencies=[Depends(require_api_key)])
async def api_import_public(request: Request):
    return await api_album_import_url(request)

@api.delete("/tracks/{track_id}", dependencies=[Depends(require_api_key)])
async def api_delete_public(track_id: str):
    return await api_album_delete(track_id)

# ---- Secure streaming (accepts either signed URL OR API key) ----
@api.get("/stream/{filename}")
def api_stream_file(
    filename: str,
    request: Request,
    sig: Optional[str] = Query(None),
    exp: Optional[int] = Query(None),
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None),
):
    """
    Secure streaming of an audio file.
    - EITHER present a valid signed URL (?sig=...&exp=...)  (recommended for vendors / playlists)
    - OR present a valid API key (X-API-Key or ?api_key=...), if you prefer key-only access.
    Supports HTTP Range for streaming/seek.
    """
    authed = False
    if sig and exp:
        try:
            exp_int = int(exp)
        except Exception:
            exp_int = 0
        if verify_stream_sig(filename, exp_int, sig):
            authed = True

    if not authed:
        token = (x_api_key or api_key or "").strip()
        if API_KEYS and token in API_KEYS:
            authed = True

    if not authed:
        raise HTTPException(status_code=401, detail="Unauthorized")

    path = os.path.join(ALBUM_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="file_not_found")

    ctype, _ = mimetypes.guess_type(filename)
    if not ctype:
        ctype = "application/octet-stream"

    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")
    if not range_header:
        def iter_all():
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 256)
                    if not chunk:
                        break
                    yield chunk
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
        return StreamingResponse(iter_all(), media_type=ctype, headers=headers)

    # Parse Range: bytes=start-end
    try:
        units, rng = range_header.split("=")
        if units.strip() != "bytes":
            raise ValueError
        start_s, end_s = rng.split("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        if start < 0 or end >= file_size or start > end:
            raise ValueError
    except Exception:
        hdrs = {"Content-Range": f"bytes */{file_size}"}
        return Response(status_code=416, headers=hdrs)

    length = end - start + 1
    def iter_range(s=start, e=end):
        with open(path, "rb") as f:
            f.seek(s)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 256, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iter_range(), status_code=206, media_type=ctype, headers=headers)

# ---- Playlist (returns signed URLs) ----
@api.get("/playlist.m3u", response_class=PlainTextResponse, dependencies=[Depends(require_api_key)])
def api_playlist_m3u(request: Request, ttl: int = Query(3600, ge=60, le=86400)):
    """
    Returns an M3U playlist of all album tracks (newest first) with
    short-lived signed streaming URLs (default 1 hour).
    Vendors can open this URL directly in players.
    """
    items = list(reversed(album_read()))  # newest first
    base = str(request.base_url).rstrip("/")  # e.g., https://tingoaiapp.onrender.com
    now = int(time.time())
    lines = ["#EXTM3U"]

    if not STREAM_SECRET:
        logger.warning("PUBLIC_STREAM_SECRET not set — playlist will not be signed securely.")

    for t in items:
        rel = t.get("url") or ""
        if not rel:
            continue
        filename = rel.rsplit("/", 1)[-1]
        title = t.get("title") or "Track"

        if STREAM_SECRET:
            exp = now + ttl
            sig = sign_stream(filename, exp)
            url = f"{base}/api/v1/stream/{filename}?exp={exp}&sig={sig}"
        else:
            url = f"{base}/api/v1/stream/{filename}?api_key=YOUR_KEY_HERE"

        lines.append(f"#EXTINF:-1,{title}")
        lines.append(url)

    return "\n".join(lines) + "\n"

# Mount the router
app.include_router(api)




# ---- Import from direct audio URL (we download & save) ----
@app.post("/api/album/import_url")
async def api_album_import_url(request: Request):
    """
    JSON body: { "url": "...", "title": "optional", "artist": "optional" }
    Only use direct audio URLs you have rights to download.
    """
    try:
        body = await request.json()
        url = (body.get("url") or "").strip()
        title = body.get("title") or "Imported Track"
        artist = body.get("artist") or ""

        if not url:
            return JSONResponse(status_code=400, content={"error": "url_required"})

        local_path = stream_import_to_album(url, max_mb=64)
        public_url = f"/album/{os.path.basename(local_path)}"
        entry = {
            "id": uuid.uuid4().hex,
            "title": title,
            "artist": artist,
            "lyrics": "",
            "file": os.path.basename(local_path),
            "url": public_url,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source": "import_url",
            "origin_url": url,
        }
        append_to_album(entry)
        return {"track": entry}
    except Exception as e:
        logger.error("Import URL error: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "import_failed", "detail": str(e)})

# ---- Delete a track from album (and file if local) ----
@app.delete("/api/album/{track_id}")
async def api_album_delete(track_id: str):
    try:
        data = album_read()
        idx = next((i for i, t in enumerate(data) if t.get("id") == track_id), None)
        if idx is None:
            return JSONResponse(status_code=404, content={"error": "not_found"})
        track = data.pop(idx)
        fn = track.get("file")
        if fn:
            path = os.path.join(ALBUM_DIR, fn)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        album_write(data)
        return {"deleted": True}
    except Exception as e:
        logger.error("Delete error: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "delete_failed", "detail": str(e)})

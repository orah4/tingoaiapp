# app.py
from fastapi import FastAPI, Form, Request, status, Response, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import requests
import os, json, uuid, datetime, logging, traceback, mimetypes, time, shutil

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
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# IMPORTANT: honor env var (e.g., /data/album on Render). Do NOT overwrite it later.
ALBUM_DIR = os.getenv("ALBUM_DIR", os.path.join(BASE_DIR, "album"))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
ALBUM_JSON = os.path.join(ALBUM_DIR, "album.json")
JOBS_JSON  = os.path.join(ALBUM_DIR, "jobs.json")   # simple persistent job store

# Create dirs/files
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(ALBUM_DIR, exist_ok=True)
for f, init in [(ALBUM_JSON, []), (JOBS_JSON, {})]:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(init, fh, indent=2)

# Static mounts (serve saved audio files under /album/<filename>)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/album", StaticFiles(directory=ALBUM_DIR), name="album")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harmonix")

# ---------- Config ----------
BARK_VERSION = "b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787"
REPLICATE_HEADERS = {
    "Authorization": f"Token {REPLICATE_API_TOKEN}",
    "Content-Type": "application/json",
}
POLL_SLEEP_SEC = 2.5
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

# ---------- Bark prediction create/poll ----------
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

# ---------- Routes ----------
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
    return {
        "album_dir": ALBUM_DIR,
        "files": list_album_files()
    }

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

# ---- Text-to-song: create job quickly; client polls /job/{id} ----
@app.post("/generate-music")
async def generate_music(
    prompt: str = Form(...),
    title: str = Form(None),
    duration: int = Form(0),
):
    try:
        lyrics = generate_lyrics(prompt)
        logger.info("Lyrics OK: %s", lyrics[:80])

        pred_id = replicate_create_prediction(lyrics)
        job_id = uuid.uuid4().hex
        created_at = datetime.datetime.utcnow().isoformat() + "Z"
        jobs = jobs_read()
        jobs[job_id] = {
            "id": job_id,
            "pred_id": pred_id,
            "status": "starting",
            "lyrics": lyrics,
            "title": title or sanitize_prompt(prompt)[:40],
            "created_at": created_at,
            "saved": False,
        }
        jobs_write(jobs)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "Job created", "job": jobs[job_id]})
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

# ---- Upload local audio into album ----
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

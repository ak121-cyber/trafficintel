# Sharing TrafficIntel with other people

The goal here is the one that actually works: **the analysis keeps running on your
GPU, and other people reach it over the internet while your PC is on.** Nothing is
uploaded to a cloud GPU, because there isn't one in this setup.

---

## Why not Streamlit Cloud (or any free host)

Worth stating plainly, because it looks like the obvious answer and isn't:

**No GPU.** Streamlit Community Cloud and similar free tiers are CPU-only.
`trafficintel.py` falls back to CPU automatically, so it won't crash — it will
just be unusable. YOLO26M at 640px on a shared CPU core runs roughly 1–3 seconds
per frame. A 76-second clip at 30fps is ~2,300 frames, so **30–90 minutes** for
something that takes a couple of minutes on a GTX 1650.

**The weights can't go in the repo.** `models/traffic_light/.../best.pt` is
167 MB, over GitHub's hard 100 MB per-file limit, which is why `.gitignore`
excludes `*.pt`. Cloud hosts deploy *from* a repo, so the weights would have to be
downloaded at runtime from separate storage — extra moving parts, and still no GPU
at the end of it.

**The dependencies are heavy.** `torch` + `paddleocr` + `paddlepaddle` together
will usually exceed a free tier's build and memory budget.

A tunnel avoids all three, because your machine keeps doing the work.

---

## Step 1 — set an access token

Do this **first**. Without it, anyone who finds the URL can upload video and spend
your GPU time and disk.

```powershell
# PowerShell
$env:TRAFFICINTEL_ACCESS_TOKEN = "paste-a-long-random-string-here"
```

```bash
# bash
export TRAFFICINTEL_ACCESS_TOKEN="paste-a-long-random-string-here"
```

Generate one rather than inventing it:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The server refuses to start on a non-loopback address without this. That refusal
is intentional — see `backend/security.py`.

Set it in the *same terminal* you start the server from. It is read at startup, so
restart after changing it.

---

## Step 2 — start the server

```bash
python backend/app.py
```

Visitors will be asked for the token at `/login`; the session then lasts 7 days in
their browser. Scripts can send it as an `X-Access-Token` header instead.

Add `--sweep` to delete leftover uploads and results from previous runs first.

---

## Step 3 — publish it with a tunnel

A tunnel gives you an HTTPS URL that forwards to your machine. **Your PC must stay
on and awake** — sleep kills the tunnel. Check your power settings.

### Cloudflare Tunnel (free, no account needed for a quick tunnel)

Install `cloudflared`, then in a second terminal:

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a `https://something-random.trycloudflare.com` URL. Share that plus the
token. The URL changes each restart; a named tunnel on a Cloudflare account gives
you a stable one.

### ngrok (alternative)

```bash
ngrok http 8000
```

Free ngrok also rotates the URL on restart.

### Note on `--host`

You do **not** need `--host 0.0.0.0` for a tunnel — `cloudflared` runs on your
machine and connects to `127.0.0.1` locally. Only use `0.0.0.0` to reach the app
from other devices on your own LAN:

```bash
python backend/app.py --host 0.0.0.0
```

Be aware this is the one case the bind check can catch. A tunnel makes the app
public *while still bound to loopback*, so the token is what protects you there,
not the bind address.

---

## What your visitors will experience

**One job at a time.** The GPU has 4 GB of VRAM, so the queue runs a single job.
Two people uploading at once means the second waits for the first to finish
completely. The API rejects a fourth queued job with HTTP 503 rather than building
a backlog nobody can see the end of.

**Uploads up to 2 GB**, and processing takes roughly as long as it does for you
locally — a few minutes per clip, longer with `--plate` OCR enabled.

**Disk is bounded.** Old finished jobs are evicted once history exceeds 50 jobs or
8 GB, and their files are deleted with them. Before this existed, trimmed jobs left
their uploads on disk permanently with no way to reach or remove them.

---

## Turning it off

Stop the tunnel process, then stop the server (Ctrl+C). The URL dies with the
tunnel. To invalidate everyone's existing sessions, change the token and restart —
the cookie holds the token itself, so a new token rejects every old cookie.

---

## If something is wrong

**Visitors see a login page they can't get past.** They have the wrong token, or
you changed it without restarting. Eight failed attempts locks that IP out for
5 minutes.

**A 401 from a script.** Send `X-Access-Token`, or export
`TRAFFICINTEL_ACCESS_TOKEN` in the shell running `tools/test_api.py` — it picks the
value up from the environment.

**Video won't play for visitors but works for you.** Almost always a stale cookie.
Have them hit `/logout` and sign in again.

**"Refusing to start."** You passed a non-loopback `--host` without a token. Set
the token, or drop the flag.

**Uptime checks.** Use `/api/ping`, which needs no token and reveals nothing.
`/api/health` is behind the gate because it exposes your device and model paths.

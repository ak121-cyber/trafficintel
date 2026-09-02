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

## Step 1 — decide who can get in

There are now two independent gates, and it is worth being clear about which one
does what.

**User accounts (always on).** Every visitor has to register and sign in before
they can reach the upload page or any result, and each account is limited to 15
credits a day — 3 videos. This is what stops one person from monopolising your
GPU, and it is enough on its own to share the app with people you know.

**A shared access token (optional).** One extra password in front of the *whole*
site, including the sign-up page, so only people you hand it to can even create an
account. Use it when the URL might be found by someone you did not invite.

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

The server refuses to start on a non-loopback address unless at least one of the
two gates is active. Accounts satisfy that, so the token is genuinely optional —
but a public bind with open registration prints a warning at startup, because
anyone who reaches the address can then sign themselves up. See
`backend/security.py`.

Set it in the *same terminal* you start the server from. It is read at startup, so
restart after changing it.

---

## Step 2 — start the server

```bash
python backend/app.py
```

MongoDB has to be running too, since that is where the accounts live. See the
README's quick start.

If a token is set, visitors are asked for it at `/login` before they see anything;
that session lasts 7 days in their browser, and they then register or sign in to
their own account as normal. Scripts can send the token as an `X-Access-Token`
header instead.

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
public *while still bound to loopback*, so the accounts — and the token, if you set
one — are what protect you there, not the bind address.

---

## What your visitors will experience

**They need an account.** The landing page is public; everything else is not. They
register with a name, email and password, and get 15 credits a day, 5 per video, so
3 videos each. The count is shown in the nav bar and on their dashboard, and it
refills the first time they use the site on a new day.

**One job at a time.** The GPU has 4 GB of VRAM, so the queue runs a single job.
Two people uploading at once means the second waits for the first to finish
completely. The API rejects a fourth queued job with HTTP 503 rather than building
a backlog nobody can see the end of.

**Uploads up to 2 GB**, and processing takes roughly as long as it does for you
locally — a few minutes per clip, longer with `--plate` OCR enabled.

**Their own history only.** Results are per-account: asking for someone else's job
id returns 404, not their video.

**Disk is bounded.** Old finished jobs are evicted once history exceeds 50 jobs or
8 GB, and their files are deleted with them. Before this existed, trimmed jobs left
their uploads on disk permanently with no way to reach or remove them.

---

## Turning it off

Stop the tunnel process, then stop the server (Ctrl+C). The URL dies with the
tunnel.

To invalidate every *login* session, change `JWT_SECRET` in `.env` and restart —
the cookies are signed with it, so a new value rejects all of them. To invalidate
the shared-token sessions, change `TRAFFICINTEL_ACCESS_TOKEN` and restart; that
cookie holds the token itself.

---

## If something is wrong

**Visitors see a login page they can't get past.** If you set a shared token, they
have the wrong one, or you changed it without restarting. Eight failed attempts
locks that IP out for 5 minutes.

**"Sign in to continue" on every action.** Their login cookie is gone or the
`JWT_SECRET` changed. Signing in again fixes it.

**"Could not connect to MongoDB."** `mongod` is not running. Accounts, credits and
history all live there, so the server will not start without it.

**A 401 from a script.** Send `X-Access-Token`, or export
`TRAFFICINTEL_ACCESS_TOKEN` in the shell running `tools/test_api.py` — it picks the
value up from the environment. `tools/test_api.py` registers its own account, so it
needs nothing else.

**Video won't play for visitors but works for you.** Almost always a stale cookie.
Have them sign out and in again.

**"Refusing to start."** You passed a non-loopback `--host` while neither gate was
available. Since accounts are always on, this now only happens if the app cannot
sign sessions at all. Drop the flag and check the startup log.

**Uptime checks.** Use `/api/ping`, which needs no token and reveals nothing.
`/api/health` also reports whether MongoDB is up, so it is useful for checking the
whole stack — but it exposes your device and model paths, so keep it behind the
shared token if you set one.

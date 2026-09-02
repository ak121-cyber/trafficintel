/* TrafficIntel — React frontend.
 *
 * Talks to the FastAPI backend on the same origin:
 *   POST /api/auth/register  create an account, sets the session cookie
 *   POST /api/auth/login     sign in
 *   GET  /api/auth/me        who am I, and how many credits are left
 *   POST /api/process        upload + toggles -> job_id (costs credits)
 *   GET  /api/status/{id}    polled until completed / failed
 *   GET  /api/history        this user's past jobs
 *   GET  /api/result/{id}/video     annotated MP4 for the player
 *   GET  /api/result/{id}/download  same file as an attachment
 *
 * The session is an HttpOnly cookie, so no token is ever held in JavaScript and
 * nothing is kept in localStorage. fetch() and XMLHttpRequest both send
 * same-origin cookies by default, which is why no request below sets a header.
 *
 * Routing is by URL hash and needs no router library. Progress is reported as
 * named stages; a percentage is shown only when the backend actually knows the
 * total frame count, otherwise the bar is indeterminate rather than faking a
 * number.
 */

const { useState, useEffect, useRef, useCallback } = React;

const POLL_MS = 1200;

const MODULES = [
  {
    key: "accident_detection",
    name: "Accident Detection",
    desc: "YOLO26M accident model combined with ByteTrack collision and motion evidence.",
    defaultOn: true,
  },
  {
    key: "traffic_light",
    name: "Traffic Light Violation",
    desc: "Traffic-light model reads the signal and measures the stop line from its own stop_line detections, then flags vehicles crossing on red.",
    defaultOn: true,
  },
  {
    key: "number_plate",
    name: "Number Plate OCR",
    desc: "Optional PaddleOCR pass on tracked vehicles. Adds significant processing time.",
    defaultOn: false,
  },
];

function fmtBytes(n) {
  if (!n && n !== 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function fmtDuration(s) {
  if (s == null) return "—";
  const total = Math.round(s);
  const m = Math.floor(total / 60);
  const sec = total % 60;
  return m > 0 ? `${m}m ${String(sec).padStart(2, "0")}s` : `${sec}s`;
}

function fmtClock(s) {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function fmtWhen(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return "—";
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

/* An id for one pending upload, used by the backend to refuse a double submit.
   crypto.randomUUID is not available on older browsers or over plain http on
   some of them, so there is a fallback - an id that is merely unlikely to
   collide is still enough to catch a double-clicked button. */
function newRequestId() {
  if (window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID().replace(/-/g, "");
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`;
}

/* Every call goes through here so that one expired session is handled in one
   place instead of at each call site. */
async function api(path, options) {
  const r = await fetch(path, options);
  let body = null;
  try {
    body = await r.json();
  } catch (e) {
    /* some responses have no body; the status still matters */
  }
  if (!r.ok) {
    const err = new Error((body && body.detail) || `Request failed (HTTP ${r.status}).`);
    err.status = r.status;
    throw err;
  }
  return body;
}

/* ---------------------------------------------------------------- Switch -- */

function Switch({ checked, onChange, disabled, label }) {
  return (
    <button
      type="button"
      role="switch"
      className="switch"
      aria-checked={checked ? "true" : "false"}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span className="knob" />
    </button>
  );
}

/* -------------------------------------------------------------- Dropzone -- */

function Dropzone({ file, onFile, onClear, disabled, sourceInfo }) {
  const [over, setOver] = useState(false);
  const inputRef = useRef(null);

  const pick = (list) => {
    if (list && list.length) onFile(list[0]);
  };

  if (file) {
    return (
      <div className="filecard">
        <div className="thumb">▶</div>
        <div className="grow">
          <div className="fname">{file.name}</div>
          <div className="fmeta">
            {fmtBytes(file.size)}
            {sourceInfo ? ` · ${sourceInfo.width}×${sourceInfo.height}` : ""}
            {sourceInfo && sourceInfo.duration_seconds
              ? ` · ${fmtDuration(sourceInfo.duration_seconds)}`
              : ""}
          </div>
        </div>
        <button className="btn small ghost" onClick={() => inputRef.current.click()} disabled={disabled}>
          Replace
        </button>
        <button className="btn small ghost" onClick={onClear} disabled={disabled}>
          Remove
        </button>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm,.m4v"
          onChange={(e) => pick(e.target.files)}
        />
      </div>
    );
  }

  return (
    <div
      className={`dropzone${over ? " over" : ""}`}
      onClick={() => !disabled && inputRef.current.click()}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) pick(e.dataTransfer.files);
      }}
    >
      <div className="icon">⬆</div>
      <div>
        <strong>Drop a traffic video here</strong> or click to browse
      </div>
      <div className="formats">MP4 · AVI · MOV · MKV · WEBM</div>
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm,.m4v"
        onChange={(e) => pick(e.target.files)}
      />
    </div>
  );
}

/* -------------------------------------------------------------- Progress -- */

// Must match the stage labels in backend/jobs.py. "Loading models" was missing,
// so during the slowest part of a cold start - loading YOLO weights onto the GPU
// - indexOf returned -1 and the rail showed no active step, which reads as a
// hung job precisely when the user is most likely to think it has hung.
const STAGE_ORDER = [
  "Uploading",
  "Queued",
  "Loading models",
  "Analyzing",
  "Finalizing",
  "Completed",
];

function StageTrack({ current, failed }) {
  const idx = STAGE_ORDER.indexOf(current);
  return (
    <div className="stages">
      {STAGE_ORDER.map((s, i) => {
        let cls = "stage";
        if (failed && s === current) cls += " failed";
        else if (i === idx) cls += " active";
        else if (idx > i) cls += " done";
        return (
          <div key={s} className={cls}>
            {i === idx && !failed && s !== "Completed" ? (
              <span className="spin" />
            ) : (
              <span>{idx > i || s === "Completed" && idx >= i ? "✓" : "·"}</span>
            )}
            {s}
          </div>
        );
      })}
    </div>
  );
}

function ProcessingPanel({ job, uploadPct, onCancel }) {
  const stage = job ? job.stage : "Uploading";
  const pct = job && job.progress != null ? job.progress * 100 : null;
  const uploading = !job;
  const shown = uploading ? uploadPct : pct;
  const indeterminate = shown == null;

  return (
    <div className="card">
      <h2>Processing Video</h2>
      <p className="hint">
        Both enabled models run in a single pass over the same video. This can take
        several minutes depending on length and resolution.
      </p>

      <StageTrack current={uploading ? "Uploading" : stage} failed={false} />

      <div className={`bar${indeterminate ? " indeterminate" : ""}`}>
        <i style={{ width: `${shown == null ? 32 : shown}%` }} />
      </div>

      <div className="progress-meta">
        <span>
          {uploading
            ? `Uploading ${uploadPct != null ? uploadPct.toFixed(0) + "%" : ""}`
            : job.total_frames > 0
            ? `frame ${job.frame.toLocaleString()} / ${job.total_frames.toLocaleString()}`
            : `frame ${job.frame.toLocaleString()}`}
        </span>
        {job && job.live && job.live.seconds != null && (
          <span>video {fmtClock(job.live.seconds)}</span>
        )}
        {job && job.live && job.live.vehicles != null && (
          <span>vehicles {job.live.vehicles}</span>
        )}
        {job && job.options && job.options.accident_detection && job.live && job.live.accidents != null && (
          <span>accidents {job.live.accidents}</span>
        )}
        {job && <span>elapsed {fmtDuration(job.elapsed_seconds)}</span>}
      </div>

      {job && (
        <div className="btn-row" style={{ marginTop: 18 }}>
          <button className="btn small ghost" onClick={onCancel}>
            Cancel processing
          </button>
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- Result -- */

function Stat({ label, value, unit, alert }) {
  return (
    <div className={`stat${alert ? " alert" : ""}`}>
      <div className="k">{label}</div>
      <div className="v">
        {value}
        {unit ? <span className="u"> {unit}</span> : null}
      </div>
    </div>
  );
}

function RedLightStatus({ s }) {
  // A violation count of 0 is ambiguous on its own: it can mean "nobody ran the
  // light" or "enforcement never armed, so nobody could have been caught". The
  // engine reports how the stop-line geometry was obtained and from which frame
  // it was live, so the difference is shown rather than left for the user to
  // guess from a reassuring-looking zero.
  const cal = s.stop_line_calibration || null;
  const source = s.stop_line_source || "none";
  const fps = s.fps || 0;
  const armedFrame = s.enforcement_active_from_frame;
  const enforcedFrames = s.enforced_frames || 0;
  const coverage =
    s.frames > 0 ? Math.round((enforcedFrames / s.frames) * 100) : null;

  if (s.red_light_enforcement_active === false) {
    // Distinguish "the model never found a stop line" from "it found one but
    // the detections contradicted each other", because the fix differs: the
    // first needs a camera angle showing the line, the second needs a manual line.
    let why;
    if (!cal) {
      why =
        "Automatic calibration was turned off for this run and no stop line was supplied.";
    } else if (cal.reason && cal.reason.indexOf("disagreed") !== -1) {
      why =
        `The model detected a stop line but the readings disagreed too much to trust ` +
        `(${cal.samples} samples). Enforcement stayed off rather than accuse vehicles ` +
        `on geometry that might be wrong.`;
    } else {
      why =
        `The model found ${cal.samples} confident stop-line detection` +
        `${cal.samples === 1 ? "" : "s"}, and ${cal.min_samples} are required before ` +
        `the geometry is trusted. The stop line is probably not visible from this angle.`;
    }
    return (
      <div className="notice info" style={{ marginTop: 16 }}>
        <strong>Red-light enforcement did not arm — the count above is not a clean bill of health</strong>
        {why} Signal state and detections are still drawn on the video, so the
        footage is usable for review.
      </div>
    );
  }

  const armedAt =
    armedFrame != null && fps > 0 ? fmtClock(armedFrame / fps) : null;

  return (
    <div className="notice ok" style={{ marginTop: 16 }}>
      <strong>
        Red-light enforcement active
        {source === "auto" ? " (auto-calibrated)" : " (stop line supplied)"}
      </strong>
      {source === "auto" && armedAt && armedFrame > 0 ? (
        <React.Fragment>
          The stop line was measured from the traffic-light detections and
          enforcement began at {armedAt}
          {coverage != null ? `, covering ${coverage}% of the clip` : ""}. Vehicles
          that crossed before that point were not checked.
        </React.Fragment>
      ) : (
        <React.Fragment>
          Enforcement covered the whole clip
          {coverage != null ? ` (${enforcedFrames} frames)` : ""}.
        </React.Fragment>
      )}
    </div>
  );
}

function Result({ job, onReset }) {
  const videoRef = useRef(null);
  const s = job.stats || {};
  const mods = s.modules || {};
  const lanes = s.lane_counts || {};
  const [playError, setPlayError] = useState(false);

  const seek = (t) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, t - 1);
      videoRef.current.play().catch(() => {});
    }
  };

  return (
    <React.Fragment>
      <div className="card">
        <div className="section-head">
          <h2 style={{ color: "var(--accent)" }}>✓ Analysis Complete</h2>
          <div className="btn-row">
            <a
              className="btn small primary"
              href={job.download_url}
              download
            >
              Download Annotated Video
            </a>
            <button className="btn small ghost" onClick={onReset}>
              Analyze another video
            </button>
          </div>
        </div>

        {job.warnings && job.warnings.length > 0 && (
          <div className="notice warn">
            <strong>Notes</strong>
            {job.warnings.map((w, i) => (
              <div key={i}>· {w}</div>
            ))}
          </div>
        )}

        {s.browser_playable === false && (
          <div className="notice warn">
            <strong>Playback may not work in this browser</strong>
            The output was written with the <code>{s.output_codec}</code> codec because an
            H.264 encoder was unavailable. Download the file and play it in VLC, or
            install an H.264-capable OpenCV/FFmpeg build.
          </div>
        )}

        {playError && (
          <div className="notice error">
            <strong>The browser could not decode this video</strong>
            Use the download button and open the file in a desktop player.
          </div>
        )}

        <video
          ref={videoRef}
          className="player"
          src={job.result_url}
          controls
          preload="metadata"
          onError={() => setPlayError(true)}
        />
      </div>

      <div className="card">
        <h2>Statistics</h2>
        <p className="hint">
          Reported by the ML pipeline for this run. Modules that were switched off
          are not shown.
        </p>

        <div className="statgrid">
          <Stat label="Unique Vehicles" value={s.unique_tracked_vehicles ?? "—"} />
          <Stat label="Lane 1" value={lanes.lane_1 ?? "—"} unit="veh" />
          <Stat label="Lane 2" value={lanes.lane_2 ?? "—"} unit="veh" />
          <Stat label="Lane 3" value={lanes.lane_3 ?? "—"} unit="veh" />
          {mods.accident_detection && (
            <Stat
              label="Accidents"
              value={s.accident_count ?? 0}
              alert={(s.accident_count || 0) > 0}
            />
          )}
          {mods.traffic_light && (
            <Stat
              label="Red-Light Violations"
              value={s.red_light_violation_count ?? 0}
              alert={(s.red_light_violation_count || 0) > 0}
            />
          )}
          {mods.number_plate && (
            <Stat label="Plates Read" value={s.number_plate_count ?? 0} />
          )}
          <Stat label="Video Duration" value={fmtDuration(s.duration_seconds)} />
          <Stat label="Processing Time" value={fmtDuration(job.elapsed_seconds)} />
        </div>

        {mods.traffic_light && <RedLightStatus s={s} />}

        {mods.accident_detection && (s.accident_events || []).length > 0 && (
          <React.Fragment>
            <div className="subhead">Accident events ({s.accident_events.length})</div>
            {/* Model-backed and motion-only events carry different weight, so the
                split is stated instead of presenting one total as if every event
                had been confirmed by the trained detector. */}
            <p className="hint" style={{ margin: "0 0 10px" }}>
              {s.accident_model_confirmed_count ?? 0} of {s.accident_events.length} confirmed
              by the trained accident model; the rest rely on collision and
              motion evidence alone. Sensitivity:{" "}
              <code>{s.accident_sensitivity || "balanced"}</code>. Click an event to
              jump to it in the video.
            </p>
            <div className="events">
              {s.accident_events.slice(0, 40).map((e, i) => (
                <div className="event seek" key={i} onClick={() => seek(e.time_seconds)}>
                  <span className="t">{fmtClock(e.time_seconds)}</span>
                  <span className="ty">{e.type}</span>
                  <span className="c">
                    {/* Motion-only incidents carry no model confidence, so the
                        engine sends null. Rendering that as "0%" would read as
                        "the model is 0% sure" rather than "the model did not
                        weigh in", so the evidence source is named instead. */}
                    {e.confidence != null
                      ? `${Math.round(e.confidence * 100)}%`
                      : "motion"}
                    {e.vehicles && e.vehicles.length
                      ? ` · ${e.vehicles.map((v) => "#" + v).join(", ")}`
                      : ""}
                  </span>
                </div>
              ))}
            </div>
          </React.Fragment>
        )}

        {mods.traffic_light && (s.red_light_violations || []).length > 0 && (
          <React.Fragment>
            <div className="subhead">
              Red-light violations ({s.red_light_violations.length})
            </div>
            <div className="events">
              {s.red_light_violations.slice(0, 40).map((v, i) => (
                <div className="event seek" key={i} onClick={() => seek(v.time_seconds)}>
                  <span className="t">{fmtClock(v.time_seconds)}</span>
                  <span className="ty">
                    Vehicle #{v.vehicle_id} crossed {v.stop_line}
                  </span>
                  <span className="c">{v.light_state}</span>
                </div>
              ))}
            </div>
          </React.Fragment>
        )}

        {mods.number_plate && (s.number_plates || []).length > 0 && (
          <React.Fragment>
            <div className="subhead">Plates ({s.number_plates.length})</div>
            <div className="events">
              {s.number_plates.slice(0, 40).map((p, i) => (
                <div className="event" key={i}>
                  <span className="t">#{p.vehicle_id}</span>
                  <span className="ty" style={{ fontFamily: "var(--mono)" }}>
                    {p.text}
                  </span>
                  <span className="c">{Math.round((p.score || 0) * 100)}%</span>
                </div>
              ))}
            </div>
          </React.Fragment>
        )}

        <details className="raw">
          <summary>Raw statistics JSON</summary>
          <pre>{JSON.stringify(s, null, 2)}</pre>
        </details>
      </div>
    </React.Fragment>
  );
}

/* ------------------------------------------------------------------ Tool -- */

/* The original single-page app, unchanged in behaviour. It is now one route
   among several, so it takes health and credits from the shell instead of
   fetching health itself, and reports a spent credit back up so the nav bar
   updates without a reload. */

function Tool({ health, credits, onCredits, onAuthLost }) {
  const [file, setFile] = useState(null);
  const [opts, setOpts] = useState(() => {
    const o = {};
    MODULES.forEach((m) => (o[m.key] = m.defaultOn));
    return o;
  });
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [uploadPct, setUploadPct] = useState(null);
  const [sourceInfo, setSourceInfo] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  // One id per pending submission, not per click. Two clicks on Process send the
  // same id, so the backend refuses the second instead of charging twice; the id
  // is cleared when a new file is chosen, so reprocessing is never blocked.
  const submitIdRef = useRef(null);

  // Capability probe: modules whose model is missing are disabled in the UI.
  useEffect(() => {
    if (!health || !health.modules) return;
    setOpts((prev) => {
      const next = { ...prev };
      if (!health.modules.traffic_light) next.traffic_light = false;
      if (!health.modules.accident_detection) next.accident_detection = false;
      if (!health.modules.number_plate) next.number_plate = false;
      return next;
    });
  }, [health]);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const poll = useCallback((jobId) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`/api/status/${jobId}`);
        if (r.status === 401) {
          // The session expired mid-job. Stop rather than poll forever, and say
          // so - the job itself keeps running on the server.
          stopPolling();
          setBusy(false);
          onAuthLost();
          return;
        }
        if (!r.ok) throw new Error("status unavailable");
        const j = await r.json();
        setJob(j);
        if (j.status === "completed" || j.status === "failed" || j.status === "cancelled") {
          stopPolling();
          setBusy(false);
          if (j.status === "failed") setError(j.error || "Processing failed.");
          if (j.status === "cancelled") setError("Processing was cancelled.");
          // This submission is settled, so retire its id. Pressing Process again
          // is then a new job and gets charged as one. Without this, cancelling
          // and retrying the same file reuses the id and the server refuses it
          // as a duplicate - a dead end, since a cancel is not refunded.
          submitIdRef.current = null;
          // A failure is refunded server-side, so re-read the balance rather
          // than assuming what it is now.
          onCredits();
        }
      } catch (e) {
        stopPolling();
        setBusy(false);
        setError("Lost contact with the backend. Is the server still running?");
      }
    }, POLL_MS);
  }, [onAuthLost, onCredits]);

  useEffect(() => stopPolling, []);

  const anyAnalysis = opts.accident_detection || opts.traffic_light;
  const canAfford = !credits || credits.can_process;

  const submit = () => {
    if (!file || busy) return;
    setError(null);
    setJob(null);
    setBusy(true);
    setUploadPct(0);

    if (!submitIdRef.current) submitIdRef.current = newRequestId();

    const fd = new FormData();
    fd.append("video", file);
    fd.append("accident_detection", opts.accident_detection ? "true" : "false");
    fd.append("traffic_light", opts.traffic_light ? "true" : "false");
    fd.append("number_plate", opts.number_plate ? "true" : "false");
    fd.append("request_id", submitIdRef.current);

    // XHR rather than fetch: it reports real upload progress.
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/process");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setUploadPct((e.loaded / e.total) * 100);
    };
    xhr.onload = () => {
      setUploadPct(null);
      let body = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch (e) {
        /* handled below */
      }
      if (xhr.status === 202 && body.job_id) {
        setSourceInfo(body.source || null);
        setJob({
          job_id: body.job_id,
          status: "queued",
          stage: "Queued",
          frame: 0,
          total_frames: (body.source && body.source.frames) || 0,
          live: {},
          elapsed_seconds: 0,
          options: body.options,
          progress: null,
          warnings: [],
        });
        onCredits();
        poll(body.job_id);
        return;
      }

      setBusy(false);
      if (xhr.status === 401) {
        onAuthLost();
        return;
      }
      // 402 means the upload was refused before anything started, so nothing was
      // charged. Re-read the balance so the number shown is the real one.
      if (xhr.status === 402) onCredits();
      setError(body.detail || `Upload failed (HTTP ${xhr.status}).`);
    };
    xhr.onerror = () => {
      setBusy(false);
      setUploadPct(null);
      setError("Could not reach the backend. Make sure the server is running.");
    };
    xhr.send(fd);
  };

  const cancel = async () => {
    if (!job) return;
    try {
      await fetch(`/api/cancel/${job.job_id}`, { method: "POST" });
    } catch (e) {
      /* the poll will surface any real problem */
    }
  };

  const reset = () => {
    stopPolling();
    setJob(null);
    setFile(null);
    setSourceInfo(null);
    setError(null);
    setBusy(false);
    setUploadPct(null);
    submitIdRef.current = null;
  };

  const done = job && job.status === "completed";
  const hw = health && health.device;

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>Analyse a video</h1>
          <p>One pass over the footage. Results stay in your history.</p>
        </div>
        <div className="device-chip">
          <span
            className={`dot ${
              !health ? "" : health.status !== "ok" ? "bad" : hw && hw.cuda_available ? "ok" : "warn"
            }`}
          />
          {!health
            ? "connecting…"
            : health.status !== "ok"
            ? "backend unreachable"
            : hw && hw.cuda_available
            ? `CUDA · ${hw.gpu_name || "GPU"}`
            : "CPU mode"}
        </div>
      </header>

      {credits && <CreditMeter c={credits} />}

      {error && (
        <div className="notice error">
          <strong>Error</strong>
          {error}
        </div>
      )}

      {!busy && !done && !canAfford && (
        <div className="notice warn">
          <strong>Out of credits for today</strong>
          You have used all {credits.videos_per_day} of today’s videos. Your
          allowance refills automatically after midnight — nothing to click, just
          come back tomorrow.
        </div>
      )}

      {!busy && !done && (
        <React.Fragment>
          <div className="card">
            <h2>Video Upload</h2>
            <p className="hint">Uploads are stored separately from the developer test videos.</p>
            <Dropzone
              file={file}
              sourceInfo={sourceInfo}
              onFile={(f) => {
                setFile(f);
                setSourceInfo(null);
                setError(null);
              }}
              onClear={reset}
              disabled={busy}
            />
          </div>

          <div className="card">
            <h2>Processing Options</h2>
            <p className="hint">
              Independent modules. Disabled models are never loaded, which keeps GPU
              memory free.
            </p>
            <div className="modules">
              {MODULES.map((m) => {
                const available = !health || !health.modules || health.modules[m.key] !== false;
                const on = !!opts[m.key];
                return (
                  <div
                    key={m.key}
                    className={`module${on ? " on" : ""}${available ? "" : " disabled"}`}
                  >
                    <Switch
                      checked={on}
                      disabled={!available || busy}
                      label={m.name}
                      onChange={(v) => setOpts((p) => ({ ...p, [m.key]: v }))}
                    />
                    <div className="grow">
                      <div className="mname">
                        {m.name}
                        {!available && <span className="badge bad">unavailable</span>}
                      </div>
                      <div className="mdesc">
                        {available
                          ? m.desc
                          : m.key === "number_plate"
                          ? "PaddleOCR is not installed in this environment."
                          : "The trained model for this module was not found on disk."}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {!anyAnalysis && (
              <div className="notice info" style={{ marginTop: 16, marginBottom: 0 }}>
                Enable at least one analysis module to process a video.
              </div>
            )}
          </div>

          <div className="btn-row">
            <button
              className="btn primary"
              onClick={submit}
              disabled={!file || !anyAnalysis || busy || !canAfford}
            >
              {canAfford && credits
                ? `Process Video · ${credits.credits_per_video} credits`
                : "Process Video"}
            </button>
            {file && (
              <span style={{ fontSize: 13, color: "var(--muted)" }}>
                {[
                  opts.accident_detection && "Accident",
                  opts.traffic_light && "Traffic Light",
                  opts.number_plate && "OCR",
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            )}
          </div>
        </React.Fragment>
      )}

      {busy && <ProcessingPanel job={job} uploadPct={uploadPct} onCancel={cancel} />}

      {done && <Result job={job} onReset={reset} />}

      <footer className="foot">
        <span>TrafficIntel · YOLO26M · ByteTrack</span>
        <span>
          {health && health.models
            ? `accident ${health.models.accident.available ? "✓" : "✗"} · traffic-light ${
                health.models.traffic_light.available ? "✓" : "✗"
              }`
            : ""}
        </span>
      </footer>
    </div>
  );
}

/* ----------------------------------------------------------- CreditMeter -- */

function CreditMeter({ c }) {
  const pct = c.credits_total ? (c.credits_remaining / c.credits_total) * 100 : 0;
  const low = c.credits_remaining < c.credits_per_video;
  return (
    <div className={`credits${low ? " low" : ""}`}>
      <div className="credits-head">
        <span className="credits-n">
          {c.credits_remaining}
          <span className="credits-of"> / {c.credits_total} credits</span>
        </span>
        <span className="credits-sub">
          {c.videos_today} of {c.videos_per_day} videos today ·{" "}
          {c.credits_per_video} credits each
        </span>
      </div>
      <div className="credits-bar">
        <div className="credits-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="credits-note">
        {low
          ? "Refills automatically after midnight."
          : `Enough for ${Math.floor(c.credits_remaining / c.credits_per_video)} more ${
              Math.floor(c.credits_remaining / c.credits_per_video) === 1 ? "video" : "videos"
            } today.`}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------- Nav -- */

function Nav({ route, user, credits, onLogout }) {
  const [open, setOpen] = useState(false);

  // Close the mobile menu on navigation, otherwise it stays open over the page
  // you just moved to.
  useEffect(() => {
    setOpen(false);
  }, [route]);

  const link = (href, label) => (
    <a
      key={href}
      href={href}
      className={`nav-link${route === href.slice(1) ? " active" : ""}`}
    >
      {label}
    </a>
  );

  return (
    <nav className="nav">
      <div className="nav-inner">
        <a className="nav-brand" href="#/">
          Traffic<span className="tick">Intel</span>
        </a>

        <button
          className="nav-toggle"
          aria-label="Menu"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          <span />
          <span />
          <span />
        </button>

        <div className={`nav-links${open ? " open" : ""}`}>
          {user ? (
            <React.Fragment>
              {link("#/dashboard", "Dashboard")}
              {link("#/tool", "Analyse")}
              {link("#/history", "History")}
              {credits && (
                <a href="#/dashboard" className="nav-credits" title="Credits left today">
                  <span className="nav-credits-n">{credits.credits_remaining}</span>
                  <span className="nav-credits-t">/ {credits.credits_total}</span>
                </a>
              )}
              <span className="nav-who" title={user.email}>
                {user.name}
              </span>
              <button className="btn ghost small" onClick={onLogout}>
                Sign out
              </button>
            </React.Fragment>
          ) : (
            <React.Fragment>
              {link("#/", "Home")}
              {link("#/login", "Sign in")}
              <a href="#/register" className="btn primary small">
                Create account
              </a>
            </React.Fragment>
          )}
        </div>
      </div>
    </nav>
  );
}

/* ------------------------------------------------------------------ Home -- */

function Home({ user, health }) {
  // Read the limits from the server rather than hardcoding them: they are
  // configurable in .env, so a figure written into this file could be wrong.
  // Until /api/health answers, the sentences below are phrased without numbers
  // instead of quoting a guess.
  const credits = (health && health.credits) || null;
  const hw = health && health.device;

  return (
    <div className="landing">
      <section className="hero">
        <span className="eyebrow">YOLO26M · ByteTrack</span>
        <h1>
          Traffic footage in.
          <br />
          <span className="tick">Evidence</span> out.
        </h1>
        <p className="lede">
          Upload a clip and get one annotated video back, with confirmed accident
          events, red-light violations and vehicle counts. One pass over the
          footage on a single GPU — no cloud, no queue of other people’s jobs.
        </p>
        <div className="hero-cta">
          {user ? (
            <a className="btn primary" href="#/tool">
              Analyse a video
            </a>
          ) : (
            <React.Fragment>
              <a className="btn primary" href="#/register">
                Create a free account
              </a>
              <a className="btn ghost" href="#/login">
                I already have one
              </a>
            </React.Fragment>
          )}
        </div>
        {hw && (
          <p className="hero-hw">
            Running on {hw.cuda_available ? hw.gpu_name || "GPU" : "CPU"}
            {credits &&
              ` · ${credits.daily} credits a day, ${credits.videos_per_day} videos`}
          </p>
        )}
      </section>

      <section className="features">
        <article className="feature">
          <h3>Accident events, not frames</h3>
          <p>
            A crash is one incident with an identity and a lifetime. Repeated
            evidence about the same collision updates that event instead of
            raising a new alarm every few seconds.
          </p>
        </article>
        <article className="feature">
          <h3>Red light means all four things</h3>
          <p>
            A violation needs a red signal, a tracked vehicle, a measured stop
            line and a directional crossing. The report says whether enforcement
            actually armed, so a count of zero is never mistaken for a clean
            intersection.
          </p>
        </article>
        <article className="feature">
          <h3>No invented numbers</h3>
          <p>
            When the trained model did not confirm an event, the confidence is
            reported as unknown rather than as a plausible-looking figure. You can
            see which findings the model agreed with.
          </p>
        </article>
        <article className="feature">
          <h3>Your history stays yours</h3>
          <p>
            Every run is recorded against your account with what it found and what
            it cost. Videos stay on the machine doing the work — nothing is
            uploaded anywhere else.
          </p>
        </article>
      </section>

      <section className="how">
        <h2>How it works</h2>
        <ol className="steps">
          <li>
            <strong>Create an account.</strong>{" "}
            {credits
              ? `You get ${credits.daily} credits every day, automatically.`
              : "You get a fresh allowance of credits every day, automatically."}
          </li>
          <li>
            <strong>Upload a clip</strong> and pick the modules you want.{" "}
            {credits
              ? `Each video costs ${credits.per_video} credits, so that is ` +
                `${credits.videos_per_day} a day.`
              : "Each video costs a few credits, so there is a daily cap."}
          </li>
          <li>
            <strong>Watch it work.</strong> Named stages, live counts, and a
            cancel button that actually stops the job.
          </li>
          <li>
            <strong>Review and download</strong> the annotated video, or come back
            to it later from your history.
          </li>
        </ol>
      </section>
    </div>
  );
}

/* --------------------------------------------------------------- AuthForm -- */

/* Login and registration are the same form with a different field list, so they
   share one component. Passwords are sent once and never stored client-side. */

function AuthForm({ mode, onDone }) {
  const isRegister = mode === "register";
  const [form, setForm] = useState({
    name: "", email: "", password: "", confirm_password: "",
  });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const set = (k) => (e) => setForm((p) => ({ ...p, [k]: e.target.value }));

  const submit = async (e) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const body = isRegister
        ? form
        : { email: form.email, password: form.password };
      const data = await api(`/api/auth/${isRegister ? "register" : "login"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      // Nothing from the response is persisted: the session is the cookie the
      // server just set, and the password object goes out of scope here.
      onDone(data);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  return (
    <div className="auth-wrap">
      <form className="auth-card" onSubmit={submit}>
        <h1>{isRegister ? "Create your account" : "Welcome back"}</h1>
        <p className="auth-sub">
          {isRegister
            ? "Free, and you get a fresh daily allowance of credits."
            : "Sign in to analyse a video and see your history."}
        </p>

        {isRegister && (
          <label>
            Name
            <input
              type="text"
              value={form.name}
              onChange={set("name")}
              autoComplete="name"
              required
              autoFocus
            />
          </label>
        )}

        <label>
          Email
          <input
            type="email"
            value={form.email}
            onChange={set("email")}
            autoComplete="email"
            required
            autoFocus={!isRegister}
          />
        </label>

        <label>
          Password
          <input
            type="password"
            value={form.password}
            onChange={set("password")}
            autoComplete={isRegister ? "new-password" : "current-password"}
            required
          />
        </label>

        {isRegister && (
          <React.Fragment>
            <label>
              Confirm password
              <input
                type="password"
                value={form.confirm_password}
                onChange={set("confirm_password")}
                autoComplete="new-password"
                required
              />
            </label>
            <p className="auth-hint">At least 8 characters.</p>
          </React.Fragment>
        )}

        {error && <div className="auth-error">{error}</div>}

        <button className="btn primary block" type="submit" disabled={busy}>
          {busy
            ? isRegister
              ? "Creating account…"
              : "Signing in…"
            : isRegister
            ? "Create account"
            : "Sign in"}
        </button>

        <p className="auth-alt">
          {isRegister ? (
            <React.Fragment>
              Already registered? <a href="#/login">Sign in</a>
            </React.Fragment>
          ) : (
            <React.Fragment>
              No account yet? <a href="#/register">Create one</a>
            </React.Fragment>
          )}
        </p>
      </form>
    </div>
  );
}

/* -------------------------------------------------------------- Dashboard -- */

function Dashboard({ user, credits, health, onAuthLost }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api("/api/history")
      .then((d) => setRows(d.history || []))
      // A 401 here means the session expired while the page was open. Say so on
      // the login page rather than rendering it as a loading failure.
      .catch((e) => (e.status === 401 ? onAuthLost() : setError(e.message)));
  }, [onAuthLost]);

  const done = (rows || []).filter((r) => r.status === "completed");
  const spent = (rows || []).reduce((n, r) => n + (r.credits_used || 0), 0);
  const hw = health && health.device;

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>Hello, {user.name}</h1>
          <p>{user.email}</p>
        </div>
        <a className="btn primary" href="#/tool">
          Analyse a video
        </a>
      </header>

      {credits && <CreditMeter c={credits} />}

      <div className="tiles">
        <div className="tile">
          <span className="tile-n">{rows ? rows.length : "—"}</span>
          <span className="tile-l">videos submitted</span>
        </div>
        <div className="tile">
          <span className="tile-n">{rows ? done.length : "—"}</span>
          <span className="tile-l">completed</span>
        </div>
        <div className="tile">
          <span className="tile-n">{rows ? spent : "—"}</span>
          <span className="tile-l">credits spent, all time</span>
        </div>
        <div className="tile">
          <span className="tile-n">
            {hw ? (hw.cuda_available ? "GPU" : "CPU") : "—"}
          </span>
          <span className="tile-l">
            {hw && hw.cuda_available ? hw.gpu_name || "CUDA" : "no CUDA device"}
          </span>
        </div>
      </div>

      {error && (
        <div className="notice error">
          <strong>Could not load your history</strong>
          {error}
        </div>
      )}

      <div className="card">
        <h2>Recent activity</h2>
        {!rows && !error && <p className="hint">Loading…</p>}
        {rows && rows.length === 0 && (
          <p className="hint">
            Nothing yet. <a href="#/tool">Upload your first video</a> — it costs{" "}
            {credits ? `${credits.credits_per_video} credits` : "credits"}.
          </p>
        )}
        {rows && rows.length > 0 && (
          <React.Fragment>
            <HistoryTable rows={rows.slice(0, 5)} />
            {rows.length > 5 && (
              <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
                <a href="#/history">See all {rows.length} runs</a>
              </p>
            )}
          </React.Fragment>
        )}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- History -- */

function HistoryTable({ rows }) {
  return (
    <div className="table-scroll">
      <table className="history">
        <thead>
          <tr>
            <th>Video</th>
            <th>When</th>
            <th>Status</th>
            <th className="num">Credits</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.job_id}>
              <td className="fname" title={r.original_filename}>
                {r.original_filename}
              </td>
              <td className="dim">{fmtWhen(r.created_at)}</td>
              <td>
                <span className={`pill ${r.status}`}>{r.status}</span>
                {r.status === "failed" && r.error && (
                  <span className="pill-note" title={r.error}>
                    {r.error.length > 60 ? `${r.error.slice(0, 60)}…` : r.error}
                  </span>
                )}
              </td>
              <td className="num">{r.credits_used}</td>
              <td className="num">
                {r.has_output ? (
                  <a
                    className="btn ghost small"
                    href={`/api/result/${r.job_id}/download`}
                  >
                    Download
                  </a>
                ) : (
                  <span className="dim">
                    {r.status === "completed" ? "file removed" : "—"}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function History({ onAuthLost }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api("/api/history")
      .then((d) => setRows(d.history || []))
      .catch((e) => (e.status === 401 ? onAuthLost() : setError(e.message)));
  }, [onAuthLost]);

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>Your videos</h1>
          <p>Every run recorded against your account.</p>
        </div>
        <a className="btn primary" href="#/tool">
          Analyse a video
        </a>
      </header>

      {error && (
        <div className="notice error">
          <strong>Could not load your history</strong>
          {error}
        </div>
      )}

      <div className="card">
        {!rows && !error && <p className="hint">Loading…</p>}
        {rows && rows.length === 0 && (
          <p className="hint">
            No videos yet. <a href="#/tool">Start with one</a>.
          </p>
        )}
        {rows && rows.length > 0 && <HistoryTable rows={rows} />}
        {rows && rows.length > 0 && (
          <p className="hint" style={{ marginTop: 16, marginBottom: 0 }}>
            Annotated videos are kept while there is room on disk; the oldest are
            removed first, which is why an older run can show “file removed”.
          </p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------- App -- */

/* The shell: session, credits, health, and hash routing. Hash routing rather
   than a router library because the whole site is four pages and the backend
   serves index.html from one static mount - real paths would 404 on reload. */

const ROUTES = ["/", "/login", "/register", "/dashboard", "/tool", "/history"];
const PROTECTED = ["/dashboard", "/tool", "/history"];

function currentRoute() {
  const raw = (window.location.hash || "#/").replace(/^#/, "");
  return ROUTES.includes(raw) ? raw : "/";
}

function App() {
  const [route, setRoute] = useState(currentRoute);
  const [user, setUser] = useState(null);
  const [credits, setCredits] = useState(null);
  const [health, setHealth] = useState(null);
  // Distinct from "no user": until the session probe finishes we do not know,
  // and rendering the login page in the meantime would flash it at someone who
  // is in fact signed in.
  const [ready, setReady] = useState(false);
  const [flash, setFlash] = useState(null);

  useEffect(() => {
    const onHash = () => setRoute(currentRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const applySession = useCallback((data) => {
    setUser(data.user);
    setCredits({
      credits_remaining: data.credits_remaining,
      credits_total: data.credits_total,
      credits_per_video: data.credits_per_video,
      videos_today: data.videos_today,
      videos_per_day: data.videos_per_day,
      can_process: data.can_process,
    });
  }, []);

  const refreshCredits = useCallback(() => {
    api("/api/auth/me")
      .then(applySession)
      .catch(() => {
        /* the 401 path is handled where the session is first established */
      });
  }, [applySession]);

  // Session probe on load. A 401 here is the normal case for a visitor, not an
  // error, so it is not surfaced.
  useEffect(() => {
    api("/api/auth/me")
      .then(applySession)
      .catch(() => setUser(null))
      .finally(() => setReady(true));

    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, [applySession]);

  const go = (path) => {
    window.location.hash = `#${path}`;
  };

  const onAuthLost = useCallback(() => {
    setUser(null);
    setCredits(null);
    setFlash("Your session expired. Please sign in again.");
    go("/login");
  }, []);

  const logout = async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch (e) {
      /* the cookie is cleared server-side; a failure here changes nothing */
    }
    setUser(null);
    setCredits(null);
    go("/");
  };

  const afterAuth = (data) => {
    applySession(data);
    setFlash(null);
    go("/dashboard");
  };

  // Signed-in users have no reason to see the auth pages.
  useEffect(() => {
    if (!ready) return;
    if (user && (route === "/login" || route === "/register")) go("/dashboard");
  }, [ready, user, route]);

  let page;
  if (!ready) {
    page = <div className="boot">Loading TrafficIntel…</div>;
  } else if (PROTECTED.includes(route) && !user) {
    // The real gate is in FastAPI - every one of these endpoints requires a
    // session, so editing this check in the browser gains nothing. This only
    // avoids showing a page that could not work.
    page = (
      <div className="app">
        <div className="notice info">
          <strong>Sign in first</strong>
          This page needs an account. <a href="#/login">Sign in</a> or{" "}
          <a href="#/register">create one</a>.
        </div>
      </div>
    );
  } else if (route === "/login" || route === "/register") {
    // key: without it React reuses one AuthForm across both routes, and a failed
    // login's error message would follow the user onto the sign-up form.
    page = <AuthForm key={route} mode={route.slice(1)} onDone={afterAuth} />;
  } else if (route === "/dashboard") {
    page = (
      <Dashboard
        user={user}
        credits={credits}
        health={health}
        onAuthLost={onAuthLost}
      />
    );
  } else if (route === "/history") {
    page = <History onAuthLost={onAuthLost} />;
  } else if (route === "/tool") {
    page = (
      <Tool
        health={health}
        credits={credits}
        onCredits={refreshCredits}
        onAuthLost={onAuthLost}
      />
    );
  } else {
    page = <Home user={user} health={health} />;
  }

  return (
    <React.Fragment>
      <Nav route={route} user={user} credits={credits} onLogout={logout} />
      {flash && (
        <div className="app">
          <div className="notice info">{flash}</div>
        </div>
      )}
      {page}
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

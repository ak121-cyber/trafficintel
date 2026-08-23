/* TrafficIntel — React frontend.
 *
 * Talks to the FastAPI backend on the same origin:
 *   POST /api/process        upload + toggles -> job_id
 *   GET  /api/status/{id}    polled until completed / failed
 *   GET  /api/result/{id}/video     annotated MP4 for the player
 *   GET  /api/result/{id}/download  same file as an attachment
 *
 * Progress is reported as named stages. A percentage is shown only when the
 * backend actually knows the total frame count; otherwise the bar is
 * indeterminate rather than faking a number.
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

/* ------------------------------------------------------------------- App -- */

function App() {
  const [health, setHealth] = useState(null);
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

  // Capability probe: modules whose model is missing are disabled in the UI.
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        setHealth(h);
        setOpts((prev) => {
          const next = { ...prev };
          if (h.modules && !h.modules.traffic_light) next.traffic_light = false;
          if (h.modules && !h.modules.accident_detection) next.accident_detection = false;
          if (h.modules && !h.modules.number_plate) next.number_plate = false;
          return next;
        });
      })
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

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
        if (!r.ok) throw new Error("status unavailable");
        const j = await r.json();
        setJob(j);
        if (j.status === "completed" || j.status === "failed" || j.status === "cancelled") {
          stopPolling();
          setBusy(false);
          if (j.status === "failed") setError(j.error || "Processing failed.");
          if (j.status === "cancelled") setError("Processing was cancelled.");
        }
      } catch (e) {
        stopPolling();
        setBusy(false);
        setError("Lost contact with the backend. Is the server still running?");
      }
    }, POLL_MS);
  }, []);

  useEffect(() => stopPolling, []);

  const anyAnalysis = opts.accident_detection || opts.traffic_light;

  const submit = () => {
    if (!file || busy) return;
    setError(null);
    setJob(null);
    setBusy(true);
    setUploadPct(0);

    const fd = new FormData();
    fd.append("video", file);
    fd.append("accident_detection", opts.accident_detection ? "true" : "false");
    fd.append("traffic_light", opts.traffic_light ? "true" : "false");
    fd.append("number_plate", opts.number_plate ? "true" : "false");

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
        poll(body.job_id);
      } else {
        setBusy(false);
        setError(body.detail || `Upload failed (HTTP ${xhr.status}).`);
      }
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
  };

  const done = job && job.status === "completed";
  const hw = health && health.device;

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>
            Traffic<span className="tick">Intel</span>
          </h1>
          <p>AI-Powered Traffic Video Intelligence</p>
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

      {error && (
        <div className="notice error">
          <strong>Error</strong>
          {error}
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
            <button className="btn primary" onClick={submit} disabled={!file || !anyAnalysis || busy}>
              Process Video
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

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

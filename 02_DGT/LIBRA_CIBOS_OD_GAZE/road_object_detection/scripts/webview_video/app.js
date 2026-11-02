// Dashboard for infer_video.py output (window.VIDEO_DATA written as data.js).
// Plays the annotated H.264 video and shows throughput/latency + a detections-per-frame
// timeline whose playhead tracks the video.

const D = window.VIDEO_DATA;
const vid = document.getElementById("vid");
const frames = D.frames;

vid.src = D.video;
document.getElementById("src").textContent =
  `${D.stats.resolution} · ${D.stats.tiles_per_frame} tiles (${D.stats.tile_size}) · src ${D.stats.source_fps}fps · stride ${D.stats.stride}`;

if (!D.playable) {
  const w = document.getElementById("warn");
  w.hidden = false;
  w.textContent = "This video is mp4v (ffmpeg transcode unavailable) and may not play here — open the .mp4 in VLC. Stats below are still valid.";
}

// --- performance panel ---
const s = D.stats;
const rows = [
  ["Pipeline FPS", s.pipeline_fps, true],
  ["Mean latency", `${s.mean_latency_ms} ms/frame`],
  ["p50 / p90 latency", `${s.p50_latency_ms} / ${s.p90_latency_ms} ms`],
  ["Per-tile inference", `${s.per_tile_inference_ms} ms`],
  ["Tiles / frame", s.tiles_per_frame],
  ["Frames processed", s.frames_processed],
  ["Detections", `${s.total_detections} (${s.detections_per_frame}/frame)`],
  ["Device", s.device],
];
document.getElementById("perf").innerHTML = rows.map(
  ([k, v, hi]) => `<div class="k">${k}</div><div class="v${hi ? " hi" : ""}">${v}</div>`
).join("") + `<div class="k">Target</div><div class="v">Jetson Orin (OD-2) — not this GPU</div>`;

// --- per-class bars ---
const cls = s.per_class;
const maxN = Math.max(1, ...Object.values(cls));
document.getElementById("classes").innerHTML = Object.entries(cls).map(
  ([k, v]) => `<div class="bar"><span class="lbl">${k}</span>` +
    `<span class="track"><span class="fill" style="width:${(100 * v / maxN).toFixed(1)}%"></span></span>` +
    `<span class="num">${v}</span></div>`
).join("") || '<div class="lbl">no detections</div>';

// --- timeline ---
const cv = document.getElementById("timeline");
const curIdx = () =>
  Math.min(frames.length - 1, Math.max(0, Math.floor(vid.currentTime * D.out_fps)));

function draw() {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const mx = Math.max(1, ...frames.map((f) => f.n));
  const bw = W / frames.length;
  ctx.fillStyle = "#3a7afe";
  frames.forEach((f, i) => {
    const bh = (f.n / mx) * (H - 6);
    ctx.fillRect(i * bw, H - bh, Math.max(1, bw - 0.5), bh);
  });
  const idx = curIdx();
  ctx.fillStyle = "#ff5252";
  ctx.fillRect(idx * bw, 0, 2, H);
}

function tick() {
  const f = frames[curIdx()];
  document.getElementById("live").textContent =
    `t=${f.t}s · frame ${f.f} · ${f.n} detections`;
  draw();
}

function sizeCanvas() { cv.width = cv.clientWidth; draw(); }

vid.addEventListener("timeupdate", tick);
vid.addEventListener("loadedmetadata", tick);
window.addEventListener("resize", sizeCanvas);
sizeCanvas();
tick();

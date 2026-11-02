// Renders the tiled-inference manifest (window.MANIFEST, written by infer_test_tiled.py).
// Each frame is drawn at natural resolution on a <canvas> overlaid on the image; CSS
// scales both together, so boxes (in natural pixel coords) always line up.

const PALETTE = [
  "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#9a6324",
];
const colorFor = (cls) => PALETTE[cls % PALETTE.length];

const DATA = window.MANIFEST;

function makeCard(im) {
  const card = document.createElement("div");
  card.className = "card";

  const frame = document.createElement("div");
  frame.className = "frame";
  const img = document.createElement("img");
  img.src = im.file;
  img.alt = im.file;
  const cv = document.createElement("canvas");
  cv.width = im.width;
  cv.height = im.height;
  frame.append(img, cv);

  const cap = document.createElement("div");
  cap.className = "cap";

  card.append(frame, cap);
  card._im = im;
  card._cv = cv;
  card._cap = cap;

  img.addEventListener("load", () => draw(card));
  if (img.complete) draw(card);
  return card;
}

function draw(card) {
  const im = card._im;
  const cv = card._cv;
  const ctx = cv.getContext("2d");
  const minConf = parseFloat(document.getElementById("conf").value);
  const showTiles = document.getElementById("tiles").checked;
  const showLabels = document.getElementById("labels").checked;
  const lw = Math.max(2, Math.round(cv.width / 400));

  ctx.clearRect(0, 0, cv.width, cv.height);

  if (showTiles) {
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = lw;
    ctx.setLineDash([12, 8]);
    im.tiles.forEach(([x0, y0, x1, y1]) =>
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0)
    );
    ctx.setLineDash([]);
  }

  const shown = im.detections.filter((d) => d.conf >= minConf);
  const fontPx = Math.max(14, Math.round(cv.width / 64));
  ctx.font = `${fontPx}px sans-serif`;
  shown.forEach((d) => {
    const [x1, y1, x2, y2] = d.xyxy;
    const c = colorFor(d.cls);
    ctx.strokeStyle = c;
    ctx.lineWidth = lw;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (showLabels) {
      const txt = `${d.name} ${d.conf.toFixed(2)}`;
      const tw = ctx.measureText(txt).width + 8;
      const th = fontPx + 6;
      ctx.fillStyle = c;
      ctx.fillRect(x1, Math.max(0, y1 - th), tw, th);
      ctx.fillStyle = "#000";
      ctx.fillText(txt, x1 + 4, Math.max(th - 5, y1 - 5));
    }
  });

  const name = im.file.split("/").pop();
  card._cap.textContent = `${name} — ${shown.length}/${im.detections.length} shown`;
}

function init() {
  document.getElementById("meta").textContent =
    `weights: ${DATA.weights} · conf≥${DATA.conf} · merge-IoU ${DATA.merge_iou} · ${DATA.images.length} images`;

  const conf = document.getElementById("conf");
  conf.value = DATA.conf;
  document.getElementById("confval").textContent = (+DATA.conf).toFixed(2);

  const grid = document.getElementById("grid");
  DATA.images.forEach((im) => grid.appendChild(makeCard(im)));

  const redraw = () => {
    document.getElementById("confval").textContent = (+conf.value).toFixed(2);
    document.querySelectorAll(".card").forEach(draw);
  };
  conf.addEventListener("input", redraw);
  document.getElementById("tiles").addEventListener("change", redraw);
  document.getElementById("labels").addEventListener("change", redraw);
}

init();

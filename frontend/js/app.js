// Frontend Car Brand Logo Classifier — vanilla JS.
// Dilayani oleh FastAPI (origin sama) atau dibuka langsung sebagai file (API di :8000).
const API_BASE = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const resultArea = $("resultArea");
const preview = $("preview");
const topLabel = $("topLabel");
const topConfidence = $("topConfidence");
const topkList = $("topkList");
const message = $("message");
const apiStatus = $("apiStatus");

const MAX_MB = 10;
const ALLOWED = ["image/png", "image/jpeg"];
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const pct = (x) => `${(x * 100).toFixed(1)}%`;

// ---------- Status & metadata ----------
async function checkHealth() {
  try {
    const data = await (await fetch(`${API_BASE}/health`)).json();
    if (data.model_ready) {
      apiStatus.textContent = "Model Siap";
      apiStatus.className = "Status Oke";
    } else {
      apiStatus.textContent = "Model Belum Dilatih";
      apiStatus.className = "Status Down";
    }
  } catch (e) {
    apiStatus.textContent = "API Tidak Terhubung";
    apiStatus.className = "Status Down";
  }
}

// ---------- Evaluasi ----------
function renderStats(m) {
  $("statAccuracy").textContent = pct(m.accuracy);
  $("statMacroF1").textContent = m.macro_f1.toFixed(3);
  $("statLoss").textContent = m.loss != null ? m.loss.toFixed(3) : "—";
  $("statSamples").textContent = m.num_samples;
  $("evalStats").hidden = false;
  if (m.evaluated_at) {
    $("evalSub").textContent =
      `${m.evaluated_at.replace("T", " ")}` +
      (m.fine_tuned ? " · fine-tuned" : "");
  }
  $("evalTag").textContent = m.fine_tuned ? "fine-tuned" : "";
}

function renderConfusion(m) {
  const cm = m.confusion_matrix;
  const labels = m.classes;
  const maxVal = Math.max(1, ...cm.flat());
  const table = $("cmTable");

  let head =
    '<thead><tr><th class="corner">akt \\ pred</th>' +
    labels.map((l) => `<th title="${cap(l)}">${cap(l)}</th>`).join("") +
    "</tr></thead>";

  let body = "<tbody>";
  cm.forEach((row, i) => {
    body += `<tr><td class="rowhead">${cap(labels[i])}</td>`;
    row.forEach((v, j) => {
      const alpha = v === 0 ? 0 : 0.1 + 0.9 * (v / maxVal);
      const fg = alpha > 0.55 ? "#fff" : "#111";
      const cls = v > 0 ? "cell has" : "cell";
      body += `<td class="${cls}" style="background:rgba(17,17,17,${alpha.toFixed(
        3
      )});color:${fg}" title="${cap(labels[i])} → ${cap(labels[j])}: ${v}">${v}</td>`;
    });
    body += "</tr>";
  });
  body += "</tbody>";
  table.innerHTML = head + body;
  $("cmBlock").hidden = false;
}

function renderPerClass(m) {
  const table = $("pcTable");
  const head =
    "<thead><tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1</th><th>N</th></tr></thead>";
  const body =
    "<tbody>" +
    m.per_class
      .map(
        (p) =>
          `<tr><td>${cap(p.label)}</td><td>${p.precision.toFixed(
            3
          )}</td><td>${p.recall.toFixed(3)}</td><td>${p.f1.toFixed(
            3
          )}</td><td>${p.support}</td></tr>`
      )
      .join("") +
    "</tbody>";
  table.innerHTML = head + body;
  $("pcBlock").hidden = false;
}

async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) {
      $("evalNote").innerHTML =
        "Metrik belum tersedia. Latih model lalu jalankan <code>python evaluate.py</code> untuk membuat <code>models/metrics.json</code>.";
      $("evalNote").hidden = false;
      return;
    }
    const m = await res.json();
    renderStats(m);
    renderConfusion(m);
    renderPerClass(m);
  } catch (e) {
    $("evalNote").textContent = "Tidak dapat memuat metrik evaluasi.";
    $("evalNote").hidden = false;
  }
}

// ---------- Prediksi ----------
function setMessage(text, isError = false) {
  message.textContent = text || "";
  message.className = isError ? "message error" : "message";
}

function validate(file) {
  if (!ALLOWED.includes(file.type)) {
    setMessage("Format tidak didukung. Gunakan JPG atau PNG.", true);
    return false;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    setMessage(`Ukuran melebihi ${MAX_MB} MB.`, true);
    return false;
  }
  return true;
}

function renderResult(data) {
  topLabel.textContent = cap(data.label);
  topConfidence.textContent = pct(data.confidence);
  topkList.innerHTML = (data.top_k || [])
    .map(
      (it) => `
        <li>
          <div class="row"><span class="name">${cap(it.label)}</span><span class="pct">${pct(
        it.confidence
      )}</span></div>
          <div class="bar"><span style="width:${(it.confidence * 100).toFixed(
            1
          )}%"></span></div>
        </li>`
    )
    .join("");
  resultArea.classList.add("show");
}

async function classify(file) {
  if (!validate(file)) return;
  setMessage("Memproses…");
  preview.src = URL.createObjectURL(file);

  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setMessage(err.detail || `Gagal (${res.status}).`, true);
      return;
    }
    renderResult(await res.json());
    setMessage("");
  } catch (e) {
    setMessage("Tidak dapat terhubung ke server. Pastikan backend berjalan.", true);
  }
}

// ---------- Events ----------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) classify(e.target.files[0]);
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) classify(file);
});

// ---------- Init ----------
checkHealth();
loadMetrics();

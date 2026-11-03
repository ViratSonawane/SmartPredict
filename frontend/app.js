const API_BASE = "https://smartpredict-vk1y.onrender.com";

const $ = id => document.getElementById(id);

let sensorMeta = {}, currentSeries = [], currentForecast = [];
let refreshToken = 0;          // bumps on every sensor change so stale responses are ignored
let toastTimer = null;

const fmt = n => n == null || Number.isNaN(Number(n))
    ? "—"
    : Number(n).toLocaleString(undefined, { maximumFractionDigits: 3 });

const stale = token => token != null && token !== refreshToken;

async function api(path) {
    const r = await fetch(`${API_BASE}${path}`);
    const text = await r.text();
    let d;
    try {
        d = JSON.parse(text);
    } catch {
        throw new Error(r.ok
            ? "Server returned invalid JSON"
            : `Server error (${r.status}) on ${path.split("?")[0]}`);
    }
    if (!r.ok) throw new Error(d.error || `Request failed (${r.status})`);
    return d;
}

function toast(msg, ms = 3500) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}

function svgChart(history, forecast = []) {
    const svg = $("main-chart"), W = 900, H = 300, L = 58, R = 18, T = 12, B = 30,
          plotW = W - L - R, plotH = H - T - B;
    const vals = [...history.map(p => p.value), ...forecast.map(p => p.value)].filter(Number.isFinite);
    if (!vals.length) { svg.innerHTML = ""; return; }
    let min = Math.min(...vals), max = Math.max(...vals);
    if (min === max) { min -= 1; max += 1; }
    const pad = (max - min) * .08; min -= pad; max += pad;
    const x = i => L + (i / Math.max(1, history.length + forecast.length - 1)) * plotW;
    const y = v => T + (max - v) / (max - min) * plotH;
    let out = "";
    for (let i = 0; i < 5; i++) {
        const v = min + (max - min) * i / 4, yy = y(v);
        out += `<line class="gridline" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/><text class="axis-label" x="${L - 9}" y="${yy + 3}" text-anchor="end">${fmt(v)}</text>`;
    }
    const hist = history.map((p, i) => `${x(i)},${y(p.value)}`).join(" ");
    out += `<polyline class="series-line" points="${hist}"/>`;
    if (forecast.length && history.length) {
        const all = [history[history.length - 1], ...forecast];
        const fc = all.map((p, i) => `${x(history.length - 1 + i)},${y(p.value)}`).join(" ");
        out += `<polyline class="forecast-line" points="${fc}"/>`;
    }
    out += `<text class="axis-label" x="${L}" y="${H - 5}">${history[0]?.time || ""}</text><text class="axis-label" x="${W - R}" y="${H - 5}" text-anchor="end">${forecast.length ? forecast[forecast.length - 1].time : (history.at(-1)?.time || "")}</text>`;
    svg.innerHTML = out;
}

function resetForecastPanel() {
    $("next-value").textContent = "—";
    $("forecast-time").textContent = "Run a forecast to see the next estimate.";
    ["mae", "rmse", "mape"].forEach(id => $(id).textContent = "—");
}

async function loadSummary() {
    const d = await api("/api/summary");
    sensorMeta = d.sensors;
    $("source-rows").textContent = fmt(d.source_rows);
    $("processed-rows").textContent = fmt(d.rows);
    $("sensor-count").textContent = d.sensor_count;
    $("failure-count").textContent = d.failures.length;
    $("date-window").textContent = `${d.first.slice(0, 10)} — ${d.last.slice(0, 10)}`;
    $("sensor").innerHTML = Object.entries(sensorMeta).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join("");
    $("connection").textContent = "API CONNECTED";
    $("connection").style.color = "#333";
    $("failure-body").innerHTML = d.failures.map(e =>
        `<tr><td>#${e.id}</td><td>${e.start}</td><td>${e.end}</td><td>${e.type}</td></tr>`).join("");
}

async function loadSeries() {
    const s = $("sensor").value || "TP2";
    const d = await api(`/api/series?sensor=${encodeURIComponent(s)}&limit=500`);
    currentSeries = d.points;
    $("series-title").textContent = d.label;
    $("unit-pill").textContent = d.unit;
    $("chart-range").textContent = `${d.points[0]?.time || ""} to ${d.points.at(-1)?.time || ""}`;
    $("forecast-unit").textContent = d.unit;
    svgChart(currentSeries, currentForecast);
}

async function runForecast() {
    const btn = $("run-forecast");
    btn.disabled = true; btn.textContent = "Calculating…";
    try {
        const s = $("sensor").value, model = $("model").value, h = $("horizon").value;
        const d = await api(`/api/forecast?sensor=${encodeURIComponent(s)}&model=${encodeURIComponent(model)}&horizon=${h}`);
        if ($("sensor").value !== s) return;            // sensor changed while waiting
        currentForecast = d.forecast;
        svgChart(d.history, d.forecast);                // last 24h + forecast, clearly visible
        const p = d.forecast[0];
        $("next-value").textContent = fmt(p?.value);
        $("forecast-unit").textContent = d.unit;
        $("forecast-time").textContent = p ? `First estimate · ${p.time} · ${d.model}` : "No forecast available";
        $("mae").textContent = fmt(d.metrics?.mae);
        $("rmse").textContent = fmt(d.metrics?.rmse);
        $("mape").textContent = d.metrics?.mape == null ? "—" : `${fmt(d.metrics.mape)}%`;
        $("chart-range").textContent = `${d.history[0]?.time || ""} to ${d.forecast.at(-1)?.time || ""} (last 24 h + forecast)`;
        toast(d.metrics_error ? `Forecast done; metrics unavailable: ${d.metrics_error}` : "Forecast completed");
    } catch (e) {
        console.error(e);
        toast(e.message, 6000);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Run forecast <span>↗</span>';
    }
}

async function loadEvaluation(token = null, force = false) {
    $("evaluation-body").innerHTML = '<tr><td colspan="4">Loading evaluation…</td></tr>';
    const sensor = $("sensor").value || "TP2";
    const d = await api(`/api/evaluation?sensor=${encodeURIComponent(sensor)}${force ? "&refresh=1" : ""}`);
    if (stale(token)) return;
    $("eval-description").textContent = `${d.split || "Chronological holdout"} · ${fmt(d.holdout_points || 0)} validation points`;
    $("evaluation-body").innerHTML = d.results.map(r =>
        `<tr><td>${r.model}</td><td>${fmt(r.mae)}</td><td>${fmt(r.rmse)}</td><td>${r.mape == null ? "—" : fmt(r.mape) + "%"}</td></tr>`
    ).join("") || '<tr><td colspan="4">No results available</td></tr>';
}

async function loadAnomalies(token = null) {
    $("anomaly-list").innerHTML = '<p class="muted compact">Loading anomaly analysis…</p>';
    const d = await api(`/api/anomalies?sensor=${encodeURIComponent($("sensor").value || "TP2")}`);
    if (stale(token)) return;
    $("anomaly-count").textContent = fmt(d.flagged_points);
    $("anomaly-list").innerHTML = d.events.length
        ? d.events.slice().reverse().map(e => `<div class="event-item"><b>${e.start}</b><span>Peak score ${e.peak_score}</span></div>`).join("")
        : '<p class="muted compact">No clustered statistical anomalies in the analysed series.</p>';
}

async function loadDrift(token = null) {
    $("drift-status").textContent = "Checking";
    ["drift-shift", "drift-psi", "drift-ks"].forEach(id => $(id).textContent = "—");
    const d = await api(`/api/drift?sensor=${encodeURIComponent($("sensor").value || "TP2")}`);
    if (stale(token)) return;
    const labels = {
        drift_signal: "Review signal",
        no_strong_drift_signal: "No strong signal",
        insufficient_data: "Insufficient data",
        insufficient_reference: "Insufficient data"
    };
    $("drift-status").textContent = labels[d.status] || d.status;
    $("drift-shift").textContent = fmt(d.standardized_mean_shift);
    $("drift-psi").textContent = fmt(d.psi);
    $("drift-ks").textContent = d.ks_pvalue == null ? "—" : fmt(d.ks_pvalue);
    $("drift-message").textContent = d.interpretation || d.note || "Drift check completed.";
}

async function loadAdaptation(token = null) {
    const d = await api(`/api/transfer?sensor=${encodeURIComponent($("sensor").value || "TP2")}`);
    if (stale(token)) return;
    const r10 = d.results.find(x => x.target_days === 10);
    if (r10) {
        $("adapt-local").textContent = fmt(r10.source_baseline.mae);
        $("adapt-prior").textContent = fmt(r10.adapted_baseline.mae);
        $("adaptation-note").textContent = `${r10.test_points} holdout points · ${d.experiment}`;
    }
    $("transfer-detail").textContent = d.results.map(x =>
        `${x.target_days} days: source MAE ${fmt(x.source_baseline.mae)}, level-adjusted MAE ${fmt(x.adapted_baseline.mae)}.`
    ).join("  ");
}

async function checkUpdate() {
    try {
        const d = await api("/api/update-strategy");
        $("update-decision").textContent =
            `${d.mode.replace(/_/g, " ")} · ${d.steps[3]} ${d.note}`;
    } catch (e) { toast(e.message, 6000); }
}

async function refreshAll() {
    const token = ++refreshToken;
    currentForecast = [];
    resetForecastPanel();                     // reset ONCE, up front — never after slow panels finish
    try {
        await loadSeries();
    } catch (e) {
        console.error(e); toast(e.message, 6000); return;
    }
    if (stale(token)) return;
    // Independent panels: one slow/failed panel can no longer block or wipe the others.
    [["Evaluation", loadEvaluation], ["Anomalies", loadAnomalies],
     ["Drift", loadDrift], ["Adaptation", loadAdaptation]].forEach(([name, fn]) =>
        fn(token).catch(e => { console.error(name, e); toast(`${name}: ${e.message}`, 6000); }));
}

$("run-forecast").addEventListener("click", runForecast);
$("sensor").addEventListener("change", refreshAll);
$("refresh-eval").addEventListener("click", () =>
    loadEvaluation(null, true).then(() => toast("Evaluation updated")).catch(e => toast(e.message, 6000)));
$("run-adaptation").addEventListener("click", () =>
    loadAdaptation().then(() => toast("Experiment completed")).catch(e => toast(e.message, 6000)));
$("check-update").addEventListener("click", checkUpdate);

(async () => {
    try {
        await loadSummary();
        await refreshAll();
    } catch (e) {
        $("connection").textContent = "API ERROR";
        toast(e.message, 6000);
    }
})();
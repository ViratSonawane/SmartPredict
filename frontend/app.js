const API_BASE = "https://smartpredict-vk1y.onrender.com";

const $ = id => document.getElementById(id);

let sensorMeta = {}, currentSeries = [], currentForecast = [];

const fmt = n => n == null
    ? "—"
    : Number(n).toLocaleString(undefined, {
        maximumFractionDigits: 3
    });

async function api(path) {
    const r = await fetch(`${API_BASE}${path}`);
    const d = await r.json();

    if (!r.ok) {
        throw new Error(d.error || "Request failed");
    }

    return d;
}

function toast(msg){const t=$("toast");t.textContent=msg;t.classList.add("show");setTimeout(()=>t.classList.remove("show"),2600)}
function svgChart(history,forecast=[]){
 const svg=$("main-chart"), W=900,H=300,L=58,R=18,T=12,B=30, plotW=W-L-R,plotH=H-T-B;
 const vals=[...history.map(p=>p.value),...forecast.map(p=>p.value)].filter(Number.isFinite);
 if(!vals.length){svg.innerHTML="";return}
 let min=Math.min(...vals),max=Math.max(...vals);if(min===max){min-=1;max+=1}
 const pad=(max-min)*.08;min-=pad;max+=pad;
 const x=i=>L+(i/Math.max(1,history.length+forecast.length-1))*plotW;
 const y=v=>T+(max-v)/(max-min)*plotH;
 let out="";
 for(let i=0;i<5;i++){let v=min+(max-min)*i/4, yy=y(v);out+=`<line class="gridline" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axis-label" x="${L-9}" y="${yy+3}" text-anchor="end">${fmt(v)}</text>`}
 const hist=history.map((p,i)=>`${x(i)},${y(p.value)}`).join(" ");
 const all=[history[history.length-1],...forecast];
 const fc=all.map((p,i)=>`${x(history.length-1+i)},${y(p.value)}`).join(" ");
 out+=`<polyline class="series-line" points="${hist}"/><polyline class="forecast-line" points="${fc}"/>`;
 out+=`<text class="axis-label" x="${L}" y="${H-5}">${history[0]?.time||""}</text><text class="axis-label" x="${W-R}" y="${H-5}" text-anchor="end">${forecast.length?forecast[forecast.length-1].time:(history.at(-1)?.time||"")}</text>`;
 svg.innerHTML=out;
}
async function loadSummary(){
 const d=await api("/api/summary");sensorMeta=d.sensors;
 $("source-rows").textContent=fmt(d.source_rows);$("processed-rows").textContent=fmt(d.rows);$("sensor-count").textContent=d.sensor_count;
 $("failure-count").textContent=d.failures.length;$("date-window").textContent=`${d.first.slice(0,10)} — ${d.last.slice(0,10)}`;
 $("sensor").innerHTML=Object.entries(sensorMeta).map(([k,v])=>`<option value="${k}">${v.label}</option>`).join("");
 $("connection").textContent="LOCAL API CONNECTED";$("connection").style.color="#333";
 $("failure-body").innerHTML=d.failures.map(e=>`<tr><td>#${e.id}</td><td>${e.start}</td><td>${e.end}</td><td>${e.type}</td><td class="severity">${e.severity}</td></tr>`).join("");
}
async function loadSeries(){
 const s=$("sensor").value||"TP2",d=await api(`/api/series?sensor=${encodeURIComponent(s)}&limit=500`);
 currentSeries=d.points;$("series-title").textContent=d.label;$("unit-pill").textContent=d.unit;$("chart-range").textContent=`${d.points[0]?.time||""} to ${d.points.at(-1)?.time||""}`;
 $("forecast-unit").textContent=d.unit;svgChart(currentSeries,currentForecast);
}
async function runForecast(){
 const btn=$("run-forecast");btn.disabled=true;btn.textContent="Calculating…";
 try{
  const s=$("sensor").value,model=$("model").value,h=$("horizon").value;
  const d=await api(`/api/forecast?sensor=${s}&model=${encodeURIComponent(model)}&horizon=${h}`);
  currentForecast=d.forecast;svgChart(currentSeries,currentForecast);
  const p=d.forecast[0];$("next-value").textContent=fmt(p?.value);$("forecast-unit").textContent=d.unit;
  $("forecast-time").textContent=p?`First estimate · ${p.time} · ${d.model}`:"No forecast available";
  $("mae").textContent=fmt(d.metrics?.mae);$("rmse").textContent=fmt(d.metrics?.rmse);$("mape").textContent=d.metrics?.mape==null?"—":`${fmt(d.metrics.mape)}%`;
  toast("Forecast completed");
 }catch(e){toast(e.message)}finally{btn.disabled=false;btn.innerHTML='Run forecast <span>↗</span>'}
}
async function loadEvaluation(){
 const d=await api(`/api/evaluation?sensor=${$("sensor").value||"TP2"}`);
 $("eval-description").textContent=`${d.split||"Chronological holdout"} · ${fmt(d.holdout_points||0)} validation points`;
 $("evaluation-body").innerHTML=d.results.map(r=>`<tr><td>${r.model}</td><td>${fmt(r.mae)}</td><td>${fmt(r.rmse)}</td><td>${r.mape==null?"—":fmt(r.mape)+"%"}</td></tr>`).join("")||'<tr><td colspan="4">No results available</td></tr>';
}
async function loadDriftAndTransfer(){
 const sensor=$("sensor").value||"TP2";
 const [d,t]=await Promise.all([api(`/api/drift?sensor=${sensor}`),api(`/api/transfer?sensor=${sensor}`)]);
 $("drift-status").textContent=d.status==="review"?"Review recommended":d.status;
 $("drift-detail").textContent=d.distribution_shift
   ? `Distribution shift flagged (KS statistic ${d.ks_statistic}, p=${d.ks_pvalue}). Review operating mode and recent maintenance before updating the model.`
   : `No significant distribution shift detected in this comparison. KS p-value: ${d.ks_pvalue??"—"}.`;
 $("transfer-detail").textContent=t.results.map(x=>`${x.target_days} days: source baseline MAE ${fmt(x.source_baseline.mae)}; adapted MAE ${fmt(x.adapted_baseline.mae)}.`).join("  ");
}
async function loadAnomalies(){
 const d=await api(`/api/anomalies?sensor=${$("sensor").value||"TP2"}`);
 $("anomaly-count").textContent=fmt(d.flagged_points);
 $("anomaly-list").innerHTML=d.events.length?d.events.slice().reverse().map(e=>`<div class="event-item"><b>${e.start}</b><span>Peak score ${e.peak_score}</span></div>`).join(""):'<p class="muted compact">No clustered statistical anomalies in the analysed series.</p>';
}
async function refreshAll(){
 try{await loadSeries();await Promise.all([loadEvaluation(),loadAnomalies(),loadDriftAndTransfer()]);currentForecast=[];svgChart(currentSeries,[]);$("next-value").textContent="—";$("forecast-time").textContent="Run a forecast to see the next estimate.";$("mae").textContent="—";$("rmse").textContent="—";$("mape").textContent="—";}
 catch(e){toast(e.message)}
}
$("run-forecast").addEventListener("click",runForecast);
$("sensor").addEventListener("change",refreshAll);
$("refresh-eval").addEventListener("click",()=>loadEvaluation().then(()=>toast("Evaluation updated")).catch(e=>toast(e.message)));
(async()=>{try{await loadSummary();await refreshAll()}catch(e){$("connection").textContent="API ERROR";toast(e.message)}})();

async function loadDrift(){
 const d=await api(`/api/drift?sensor=${$('sensor').value||'TP2'}`);
 $('drift-status').textContent=d.status==='drift_signal'?'Review signal':d.status==='no_strong_drift_signal'?'No strong signal':'Insufficient data';
 $('drift-shift').textContent=fmt(d.standardized_mean_shift);$('drift-psi').textContent=fmt(d.psi);
 $('drift-message').textContent=d.interpretation||d.message||'Drift check completed.';
}
async function runAdaptation(){
 try{const d=await api(`/api/adaptation?sensor=${$('sensor').value||'TP2'}&days=10`);
 $('adapt-local').textContent=fmt(d.local_only.mae);$('adapt-prior').textContent=fmt(d.historical_prior_adapted.mae);
 $('adaptation-note').textContent=`${d.test_points} holdout points · ${d.scope_warning}`;toast('Adaptation experiment completed');
 }catch(e){toast(e.message)}
}
async function checkUpdate(){
 try{const d=await api(`/api/model-update?sensor=${$('sensor').value||'TP2'}`);
 $('update-decision').textContent=`${d.decision}: ${d.policy||d.reason||''}`;toast('Shadow evaluation completed');
 }catch(e){toast(e.message)}
}
$('run-adaptation')?.addEventListener('click',runAdaptation);
$('check-update')?.addEventListener('click',checkUpdate);
const oldRefreshAll=refreshAll;
refreshAll=async function(){await oldRefreshAll();try{await loadDrift()}catch(e){toast(e.message)}};
$('sensor')?.addEventListener('change',()=>loadDrift().catch(e=>toast(e.message)));

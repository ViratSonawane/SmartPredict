from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import os

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
except ImportError:  # pragma: no cover - optional dependency for forecasting endpoints
    ExponentialSmoothing = None
    Holt = None

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
except ImportError:  # pragma: no cover - optional dependency for scoring / hybrid model
    mean_absolute_error = None
    mean_squared_error = None
    RandomForestRegressor = None

try:
    from scipy.stats import ks_2samp
except ImportError:  # pragma: no cover - optional dependency for drift detection
    ks_2samp = None


def require_statsmodels():
    if Holt is None or ExponentialSmoothing is None:
        raise RuntimeError("statsmodels is not installed. Run: pip install -r requirements.txt")


def require_sklearn():
    if mean_absolute_error is None or mean_squared_error is None or RandomForestRegressor is None:
        raise RuntimeError("scikit-learn is not installed. Run: pip install -r requirements.txt")


def require_scipy():
    if ks_2samp is None:
        raise RuntimeError("scipy is not installed. Run: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RAW = DATA / "raw" / "MetroPT3(AirCompressor).csv"

FAILURES = [
    {"id": 1, "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:00", "type": "Air leak"},
    {"id": 2, "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00", "type": "Air leak"},
    {"id": 3, "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00", "type": "Air leak"},
    {"id": 4, "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00", "type": "Air leak"},
]
SENSORS = {
    "TP2": {"label":"Compressor pressure", "unit":"bar"},
    "TP3": {"label":"Pneumatic panel pressure", "unit":"bar"},
    "H1": {"label":"Cyclonic separator pressure", "unit":"bar"},
    "DV_pressure": {"label":"Dryer discharge pressure", "unit":"bar"},
    "Reservoirs": {"label":"Reservoir pressure", "unit":"bar"},
    "Oil_temperature": {"label":"Oil temperature", "unit":"°C"},
    "Motor_current": {"label":"Motor current", "unit":"A"},
    "Caudal_impulses": {"label":"Air-flow impulses", "unit":"count"},
}
DIGITAL = ["COMP","DV_eletric","Towers","MPG","LPS","Pressure_switch","Oil_level"]
ALL_NUMERIC = list(SENSORS) + DIGITAL

# In-process caches so every endpoint does not re-read the CSV / recompute heavy results.
_FRAME_CACHE = {}
_EVAL_CACHE = {}


def _load_frame_uncached(interval="5min"):
    freq = "1min" if interval == "1min" else "5min"

    path = PROCESSED / (
        "metropt_1min.csv" if freq == "1min"
        else "metropt_5min.csv"
    )

    # OPTION 1: Use a local processed dataset if available.
    if path.exists():
        print("Loading local processed dataset...")

        df = pd.read_csv(
            path,
            parse_dates=["timestamp"],
            index_col="timestamp"
        )

    # OPTION 2: Download the processed dataset from Hugging Face.
    elif freq == "5min" and os.getenv("HF_DATASET_REPO"):
        from huggingface_hub import hf_hub_download

        repo_id = os.environ["HF_DATASET_REPO"]

        filename = os.getenv(
            "HF_DATASET_FILENAME",
            "metropt_5min.csv"
        )

        print(f"Downloading dataset from Hugging Face: {repo_id}")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN") or None
        )

        print("Dataset downloaded successfully.")

        df = pd.read_csv(
            downloaded_path,
            parse_dates=["timestamp"],
            index_col="timestamp"
        )

    # OPTION 3: Use the original raw dataset as a fallback.
    else:
        if not RAW.exists():
            raise FileNotFoundError(
                "Dataset missing. Configure HF_DATASET_REPO "
                "on Render or provide the raw CSV locally."
            )

        print("Processing raw MetroPT-3 dataset...")

        df = pd.read_csv(
            RAW,
            usecols=lambda c: c == "timestamp" or c in ALL_NUMERIC,
            parse_dates=["timestamp"]
        )

        df = (
            df.sort_values("timestamp")
            .set_index("timestamp")
            .resample(freq)
            .mean()
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index_label="timestamp")

    df = df.sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def load_frame(interval="5min"):
    key = "1min" if interval == "1min" else "5min"
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = _load_frame_uncached(interval)
    return _FRAME_CACHE[key]


def to_points(index, values, ndigits=5):
    """Build [{time, value}] and drop NaN/inf: Flask would emit invalid JSON 'NaN' otherwise."""
    out = []
    for t, v in zip(index, values):
        v = float(v)
        if np.isfinite(v):
            out.append({"time": t.strftime("%Y-%m-%d %H:%M"), "value": round(v, ndigits)})
    return out


def clean_series(series):
    s = pd.to_numeric(series, errors="coerce").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 3:
        return s.dropna()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="5min")
    s = s.reindex(full_idx)
    return s.interpolate(method="time", limit=2, limit_area="inside")

def _regularize_model_series(series):
    s = pd.to_numeric(series, errors="coerce").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if isinstance(s.index, pd.DatetimeIndex) and len(s):
        index = pd.date_range(s.index.min(), s.index.max(), freq="5min")
        s = s.reindex(index).interpolate(method="time", limit_area="inside")
    return s.dropna().astype(float)

def safe_sensor(value):
    return value if value in SENSORS else "TP2"

def metric_pack(actual, predicted):
    require_sklearn()
    a = np.asarray(actual, dtype=float); p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if not len(a): return {"mae":None, "rmse":None, "mape":None}
    mae = float(mean_absolute_error(a,p))
    rmse = float(np.sqrt(mean_squared_error(a,p)))
    nz = np.abs(a) > 1e-8
    mape = float(np.mean(np.abs((a[nz]-p[nz])/a[nz]))*100) if nz.any() else None
    return {"mae":round(mae,5), "rmse":round(rmse,5),
            "mape":round(mape,3) if mape is not None else None}

def _holt(y, horizon):
    require_statsmodels()
    return np.asarray(Holt(y, initialization_method="estimated").fit(optimized=True).forecast(horizon), dtype=float)

def _hw(y, horizon):
    require_statsmodels()
    if len(y) >= 4*288:
        model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=288,
                                    initialization_method="estimated")
        return np.asarray(model.fit(optimized=True).forecast(horizon), dtype=float)
    return _holt(y, horizon)

def _prophet(y, horizon):
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise RuntimeError("Prophet is not installed. Run: pip install prophet") from exc
    frame = pd.DataFrame({"ds": y.index, "y": y.values})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=False, uncertainty_samples=0)
    model.fit(frame)
    future = pd.DataFrame({"ds": pd.date_range(y.index[-1] + pd.Timedelta(minutes=5),
                                                periods=horizon, freq="5min")})
    return np.asarray(model.predict(future)["yhat"], dtype=float)

def _hybrid(y, horizon):
    # Additive seasonal/trend baseline plus a small random forest trained on
    # one-step residuals. Recursive residual prediction adds nonlinear correction.
    require_statsmodels()
    require_sklearn()
    base = _hw(y, horizon)
    fitted = _hw_fitted(y)
    residual = y.to_numpy()[-len(fitted):] - fitted
    if len(residual) < 100:
        return base
    lags = 12
    X, target = [], []
    for i in range(lags, len(residual)):
        X.append(residual[i-lags:i])
        target.append(residual[i])
    if len(X) < 50:
        return base
    model = RandomForestRegressor(n_estimators=40, max_depth=5, min_samples_leaf=4,
                                  random_state=42, n_jobs=1)
    model.fit(np.asarray(X), np.asarray(target))
    history = list(residual[-lags:])
    correction = []
    for _ in range(horizon):
        pred = float(model.predict(np.asarray(history[-lags:]).reshape(1,-1))[0])
        correction.append(pred)
        history.append(pred)
    return base + np.asarray(correction)

def _hw_fitted(y):
    require_statsmodels()
    if len(y) >= 4*288:
        try:
            fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=288,
                                       initialization_method="estimated").fit(optimized=True)
            return np.asarray(fit.fittedvalues, dtype=float)
        except Exception:
            pass
    fit = Holt(y, initialization_method="estimated").fit(optimized=True)
    return np.asarray(fit.fittedvalues, dtype=float)

def forecast_values(y, horizon, model):
    y = y.iloc[-21*24*12:]
    if y.dropna().shape[0] < 100:
        raise ValueError("At least 100 valid observations are required in the latest 21 days.")
    y = _regularize_model_series(y)
    if len(y) < 100:
        raise ValueError("At least 100 regularized observations are required.")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one interval.")
    if model == "Holt trend": pred = _holt(y,horizon)
    elif model == "Holt-Winters": pred = _hw(y,horizon)
    elif model == "Prophet": pred = _prophet(y,horizon)
    elif model == "Hybrid": pred = _hybrid(y,horizon)
    elif model == "Seasonal naive":
        period=min(288,len(y)); pred=np.asarray([y.iloc[-period+(i%period)] for i in range(horizon)])
    else: raise ValueError("Unknown forecasting model.")
    return pred

def _evaluation_window(y):
    y = pd.to_numeric(y, errors="coerce").sort_index()
    y = y[~y.index.duplicated(keep="last")]
    if y.empty:
        return None
    if isinstance(y.index, pd.DatetimeIndex):
        index = pd.date_range(y.index.min(), y.index.max(), freq="5min")
        y = y.reindex(index)
    if y.dropna().shape[0] < 150:
        return None
    recent = y.iloc[-min(len(y),14*24*12):]
    split = max(100,int(len(recent)*.8))
    train, test = recent.iloc[:split], recent.iloc[split:]
    if len(test) == 0 or test.notna().sum() == 0:
        return None
    return train, test

def evaluate_model(y, model):
    window = _evaluation_window(y)
    if window is None:
        return None
    train, test = window
    if model == "Seasonal naive":
        train = _regularize_model_series(train)
        period=min(288,len(train))
        pred=np.asarray([train.iloc[-period+(i%period)] for i in range(len(test))])
    else:
        # Forecast the entire regular holdout; score only timestamps with observations.
        pred=forecast_values(train,len(test),model)
    return metric_pack(test.to_numpy(),pred)

def robust_anomalies(s, threshold=4.5):
    s=s.dropna().astype(float)
    window=min(144,max(24,len(s)//20))
    med=s.rolling(window,center=True,min_periods=max(12,window//3)).median()
    dev=(s-med).abs()
    mad=dev.rolling(window,center=True,min_periods=max(12,window//3)).median()
    score=(dev/(1.4826*mad.replace(0,np.nan))).replace([np.inf,-np.inf],np.nan)
    flagged=score[score>threshold].dropna()
    groups=[]; group=[]
    for ts,val in flagged.items():
        if group and ts-group[-1][0]>pd.Timedelta(minutes=15):
            groups.append(group); group=[]
        group.append((ts,float(val)))
    if group: groups.append(group)
    return flagged, groups

def event_validation(groups):
    # Event overlap is descriptive only: broad source windows are not row-level labels.
    output=[]
    for event in FAILURES:
        start=pd.Timestamp(event["start"]); end=pd.Timestamp(event["end"])
        count=sum(1 for g in groups if g[0][0] <= end and g[-1][0] >= start)
        output.append({**event,"overlapping_anomaly_groups":count,
                       "overlap_detected":bool(count)})
    return output

def drift_report(s):
    require_scipy()
    s=s.dropna().astype(float)
    empty={"standardized_mean_shift":None,"psi":None,"ks_statistic":None,"ks_pvalue":None,
           "distribution_shift":None}
    if len(s)<100:
        return {"status":"insufficient_data","interpretation":"Not enough observations for a drift check.",**empty}
    recent=s.iloc[-min(288,len(s)):]
    reference=s.iloc[-min(2880,len(s)):-min(288,len(s))]
    if len(reference)<50:
        return {"status":"insufficient_reference","interpretation":"Not enough reference history for a drift check.",**empty}

    # 1) Standardized mean shift (recent mean vs reference mean, in reference std units).
    ref_std=float(reference.std()) or 1e-9
    mean_shift=float((recent.mean()-reference.mean())/ref_std)

    # 2) Population Stability Index on reference deciles.
    edges=np.unique(np.quantile(reference.to_numpy(),np.linspace(0,1,11)))
    if len(edges)>=3:
        edges[0],edges[-1]=-np.inf,np.inf
        r=np.clip(np.histogram(reference,edges)[0]/len(reference),1e-4,None)
        c=np.clip(np.histogram(recent,edges)[0]/len(recent),1e-4,None)
        psi=float(np.sum((c-r)*np.log(c/r)))
    else:
        psi=0.0

    # 3) KS test: descriptive, not a causal diagnosis.
    ks=ks_2samp(reference.to_numpy(),recent.to_numpy())
    shifted=bool(ks.pvalue<0.01)
    signal=bool(shifted or psi>0.25)
    return {"status":"drift_signal" if signal else "no_strong_drift_signal",
            "reference_points":len(reference),"recent_points":len(recent),
            "standardized_mean_shift":round(mean_shift,4),"psi":round(psi,4),
            "ks_statistic":round(float(ks.statistic),4),"ks_pvalue":round(float(ks.pvalue),6),
            "distribution_shift":shifted,
            "interpretation":("Distribution shift flagged. Review operating mode and recent maintenance before updating the model."
                              if signal else
                              "No strong distribution shift between the recent window and the reference window."),
            "note":"KS test and PSI compare the latest 24 h with the preceding reference window; descriptive only."}

app=Flask(__name__,static_folder="../frontend",static_url_path="")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get(
        "FRONTEND_ORIGIN", "*"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.get("/")
def home(): return send_from_directory(app.static_folder,"index.html")
@app.get("/api/health")
def health(): return jsonify({"status":"ok","project":"SmartPredict","dataset":"MetroPT-3"})
@app.get("/api/summary")
def summary():
    try:
        df=load_frame("5min"); valid=df.dropna(how="all")
        raw_rows=1516948
        return jsonify({"rows":int(len(valid)),"first":str(valid.index.min()),"last":str(valid.index.max()),
                        "sensor_count":len(SENSORS),"sensors":SENSORS,"failures":FAILURES,
                        "source_rows":raw_rows,"sampling":"5-minute means from supplied source CSV"})
    except Exception as e: return jsonify({"error":str(e)}),500
@app.get("/api/series")
def series():
    try:
        sensor=safe_sensor(request.args.get("sensor","TP2")); df=load_frame("5min")
        s=clean_series(df[sensor]).dropna()
        limit=int(np.clip(int(request.args.get("limit",500)),50,2000))
        if len(s)>limit: s=s.iloc[np.linspace(0,len(s)-1,limit).astype(int)]
        return jsonify({"sensor":sensor,"label":SENSORS[sensor]["label"],"unit":SENSORS[sensor]["unit"],
                        "points":to_points(s.index,s.to_numpy())})
    except Exception as e: return jsonify({"error":str(e)}),500
@app.get("/api/forecast")
def forecast():
    try:
        sensor=safe_sensor(request.args.get("sensor","TP2")); model=request.args.get("model","Holt trend")
        horizon=int(np.clip(int(request.args.get("horizon",24)),1,72)); s=clean_series(load_frame("5min")[sensor])
        pred=forecast_values(s,horizon,model)
        idx=pd.date_range(s.dropna().index[-1]+pd.Timedelta(minutes=5),periods=len(pred),freq="5min")
        forecast_points=to_points(idx,pred)
        if not forecast_points:
            raise ValueError("The model produced no finite forecast values.")
        metrics_error = None
        try:
            metric=evaluate_model(s,model) or {"mae":None,"rmse":None,"mape":None}
        except Exception as exc:
            metric={"mae":None,"rmse":None,"mape":None}
            metrics_error=str(exc)
        history=s.dropna().iloc[-288:]
        return jsonify({"sensor":sensor,"label":SENSORS[sensor]["label"],"unit":SENSORS[sensor]["unit"],
                        "model":model,"horizon":horizon,
                        "history":to_points(history.index,history.to_numpy()),
                        "forecast":forecast_points,
                        "metrics":metric,"metrics_error":metrics_error,
                        "note":"Internal missing intervals are time-interpolated for model fitting; evaluation scores only observed holdout values. Sensor-value forecasts are not calibrated failure probabilities."})
    except Exception as e: return jsonify({"error":str(e)}),400
@app.get("/api/anomalies")
def anomalies():
    try:
        sensor=safe_sensor(request.args.get("sensor","TP2")); s=clean_series(load_frame("5min")[sensor])
        flagged,groups=robust_anomalies(s)
        events=[{"start":g[0][0].strftime("%Y-%m-%d %H:%M"),"end":g[-1][0].strftime("%Y-%m-%d %H:%M"),
                 "peak_score":round(max(x[1] for x in g),2),"points":len(g)} for g in groups[-30:]]
        return jsonify({"sensor":sensor,"threshold":4.5,"flagged_points":int(len(flagged)),
                        "events":events,"reported_failures":FAILURES,
                        "event_validation":event_validation(groups),
                        "method":"Rolling median and median absolute deviation (MAD), robust threshold 4.5.",
                        "caution":"Anomaly flags are not confirmed failures; reference windows are event-level context."})
    except Exception as e: return jsonify({"error":str(e)}),500
@app.get("/api/evaluation")
def evaluation():
    try:
        sensor=safe_sensor(request.args.get("sensor","TP2"))
        # Heavy (Prophet + Hybrid): serve from cache unless ?refresh=1 is passed.
        if request.args.get("refresh")!="1" and sensor in _EVAL_CACHE:
            return jsonify(_EVAL_CACHE[sensor])
        s=clean_series(load_frame("5min")[sensor])
        window=_evaluation_window(s)
        models=["Holt trend","Holt-Winters","Prophet","Hybrid","Seasonal naive"]
        results=[]
        for model in models:
            try:
                result=evaluate_model(s,model)
                if result is not None: results.append({"model":model,**result})
            except Exception as exc:
                results.append({"model":model,"unavailable":str(exc),"mae":None,"rmse":None,"mape":None})
        payload={"sensor":sensor,"unit":SENSORS[sensor]["unit"],
                 "split":"Chronological 80/20 holdout on the latest 14 days; no random shuffle.",
                 "holdout_points":int(window[1].notna().sum()) if window else 0,
                 "results":results}
        _EVAL_CACHE[sensor]=payload
        return jsonify(payload)
    except Exception as e: return jsonify({"error":str(e)}),500
@app.get("/api/drift")
def drift():
    try:
        sensor=safe_sensor(request.args.get("sensor","TP2"))
        return jsonify({"sensor":sensor,**drift_report(clean_series(load_frame("5min")[sensor]))})
    except Exception as e: return jsonify({"error":str(e)}),500
@app.get("/api/transfer")
def transfer():
    # One dataset / one compressor only: this is a time-based proxy experiment, NOT
    # cross-machine validation. Earlier history is source knowledge; latest 10/15 days
    # are target calibration windows. Baseline and offset-adapted Holt are compared.
    try:
        sensor=safe_sensor(request.args.get("sensor","TP2")); s=clean_series(load_frame("5min")[sensor]).dropna()
        n10=10*24*12; n15=15*24*12
        target=s.iloc[-n15:]
        source=s.iloc[:-n15]
        if len(source)<100 or len(target)<n10: raise ValueError("Not enough history for the 10–15 day temporal proxy experiment.")
        rows=[]
        for days,n in [(10,n10),(15,n15)]:
            target_window=target.iloc[-n:]
            calibration=target_window.iloc[:max(1,int(n*.2))]
            test=target_window.iloc[len(calibration):]
            source_level=float(source.iloc[-min(len(source),288):].median())
            target_level=float(calibration.median())
            # Transfer adjustment is a transparent level-shift calibration, not learned fleet transfer.
            adjustment=target_level-source_level
            source_pred=np.repeat(source_level,len(test))
            adapted_pred=np.repeat(source_level+adjustment,len(test))
            rows.append({"target_days":days,"calibration_points":len(calibration),"test_points":len(test),
                         "source_baseline":metric_pack(test,source_pred),
                         "adapted_baseline":metric_pack(test,adapted_pred)})
        return jsonify({"sensor":sensor,"experiment":"Temporal source-to-target proxy on a single asset",
                        "results":rows,"limitation":"MetroPT-3 supplied here contains one APU time series, not 120 machine IDs. This experiment does not establish cross-machine transfer learning."})
    except Exception as e: return jsonify({"error":str(e)}),400
@app.get("/api/update-strategy")
def update_strategy():
    # Operational guidance endpoint: candidate model is only promoted after a shadow
    # comparison; no live model is silently replaced by this academic demo.
    return jsonify({"mode":"shadow_evaluation","steps":[
        "Keep the incumbent model serving predictions.",
        "Fit a candidate model on a recent chronological training window.",
        "Evaluate candidate and incumbent on the same untouched holdout.",
        "Promote only when candidate MAE improves by at least 5% and RMSE does not materially worsen.",
        "Retain incumbent artefact and rollback if post-promotion error breaches its control limit."
    ],"automatic_promotion":False,"note":"Promotion is intentionally manual in this academic prototype; no production machine is controlled."})
@app.get("/api/failures")
def failures(): return jsonify({"events":FAILURES,"source":"MetroPT-3 dataset description"})
if __name__=="__main__":
    app.run(host="127.0.0.1",port=5000,debug=False)
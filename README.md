# SmartPredict — Edge-Based Predictive Maintenance

A local academic prototype based on the MetroPT-3 Air Production Unit (APU) dataset. The interface is intentionally minimal and uses black, white and grey only.

## Included dataset

- Full supplied CSV: `data/raw/MetroPT3(AirCompressor).csv`
- Source description PDF: `data/raw/Data Description_Metro.pdf`
- Derived 1-minute and 5-minute mean datasets: `data/processed/`
- Dataset inventory: `data/dataset_inventory.json`

The supplied CSV contains 1,516,948 records, with timestamps from 2020-02-01 00:00:00 to 2020-09-01 03:59:50. The dataset description reports 15,169,480 instances and 1 Hz, while the provided CSV contains 1,516,948 rows at approximately 10-second intervals. The project uses the actual supplied CSV for calculations and records this source discrepancy transparently.

## Implemented functionality

1. Sensor history and selectable channels (pressure, oil temperature, motor current and airflow impulses).
2. Short-horizon forecasting: Holt trend, Holt-Winters daily seasonality, seasonal-naive baseline, Holt-plus-median-residual hybrid, and Prophet.
3. Chronological holdout model comparison using MAE, RMSE and MAPE. Internal missing intervals are time-interpolated for model fitting so the 5-minute time grid stays regular; missing target observations are excluded from scoring. Model-specific failures are returned as unavailable rather than silently omitted.
4. Robust anomaly screening using a rolling median and rolling median absolute deviation (MAD).
5. Coarse event-window overlap against the four documented air-leak periods. This is not row-level classification accuracy because the source does not supply row-level labels.
6. Sensor-distribution drift screening using standardized mean shift and Population Stability Index (PSI). A signal requests review; it does not trigger automatic retraining.
7. A 10-day limited-history temporal warm-start experiment comparing local-only fitting with a prior-initialized model. This is a same-series proxy, not cross-machine transfer learning.
8. Shadow model-update recommendation comparing candidate models. It never deploys or overwrites a production model; engineering approval remains required.
9. Local Flask inference on `127.0.0.1`, so the prototype can continue operating without cloud connectivity after dependencies are installed.

## Install and run on Windows

1. Install 64-bit Python 3.10 or 3.11 and ensure the Python Launcher (`py`) is available.
2. Extract the ZIP and open Command Prompt in the `SmartPredict` folder.
3. Run `run_windows.bat`. It creates a virtual environment, installs core requirements and Prophet, then starts the local server.
4. Open `http://127.0.0.1:5000`.

The dashboard uses the same-origin API by default. If hosting the frontend separately, set `window.SMARTPREDICT_API_BASE` to the backend origin before loading `frontend/app.js`.

Manual setup:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-prophet.txt
python backend\app.py
```

If Prophet cannot be installed on a particular Windows/Python combination, the rest of the application remains usable; Prophet will be reported as unavailable until its optional dependency is installed.

## API endpoints

- `GET /api/health`
- `GET /api/summary`
- `GET /api/series?sensor=TP2`
- `GET /api/forecast?sensor=TP2&model=Prophet&horizon=24`
- `GET /api/evaluation?sensor=TP2`
- `GET /api/anomalies?sensor=TP2`
- `GET /api/drift?sensor=TP2`
- `GET /api/adaptation?sensor=TP2&days=10`
- `GET /api/model-update?sensor=TP2`
- `GET /api/failures`

## Important scope and limitations

- MetroPT-3 represents one compressor/APU, not 120 independently identified machines across five factories. The included adaptation experiment is explicitly a temporal warm-start proxy on one time series. Genuine cross-machine transfer learning requires machine IDs and historical records from multiple machines.
- The source describes four air-leak event windows, not per-observation fault labels. Event overlap is only a coarse contextual check; precision, recall, F1 and calibrated failure probability cannot be claimed from these labels.
- The dataset does not provide every sensor in the case study (for example, it does not provide a dedicated vibration channel or operating-speed channel in this project’s selected feature set).
- Edge inference is demonstrated by local execution and offline operation, not certified industrial gateway deployment. Hardware latency, memory and power measurements would require the target gateway.
- Forecasting sensor values and flagging unusual behaviour are not the same as predicting failure probability or remaining useful life.
- Drift monitoring uses sensor-distribution shifts. Prediction-error monitoring should be fed with newly observed actual values in a production deployment; the included chronological evaluation supplies offline error metrics.

## Project structure

```text
SmartPredict/
├── backend/app.py
├── frontend/index.html
├── frontend/style.css
├── frontend/app.js
├── data/raw/                 # Full source CSV and PDF
├── data/processed/           # Full-history aggregates
├── data/dataset_inventory.json
├── reports/Project_Report.md
├── notebooks/quick_analysis.py
├── requirements.txt
├── requirements-prophet.txt
└── run_windows.bat
```

## Dataset reference

MetroPT-3, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset

Davari et al. (2021), *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry*, IEEE DSAA.

# SmartPredict: Edge-Based Predictive Maintenance Using Time-Series Forecasting

## 1. Abstract
SmartPredict is a local predictive-maintenance prototype using the MetroPT-3 Air Production Unit (APU) dataset. It combines sensor-history visualization, statistical forecasting, robust anomaly screening, offline inference, chronological model evaluation, drift screening and a shadow-update recommendation. It demonstrates an implementable subset of the integrated framework requested in the case study while clearly identifying what the supplied single-machine dataset cannot validate.

## 2. Problem statement and alignment
The case study describes 120 machines across five factories. Established machines have three to five years of data, while new machines have only 10–15 days. It requires low-latency local inference, operation during connectivity loss, handling recurring and irregular sensor behaviour, adaptation to changing conditions, resource-conscious deployment, and safe model maintenance.

MetroPT-3 is relevant because it contains real industrial compressor sensor readings and documented air-leak event windows. It represents one compressor/APU, however, and does not provide independent identities and multi-year histories for 120 machines. Therefore, this implementation demonstrates forecasting and monitoring on one compressor. It does not claim to have trained or validated a fleet-wide transfer-learning model.

## 3. Dataset and integrity
The supplied CSV contains 1,516,948 observations from 2020-02-01 00:00:00 to 2020-09-01 03:59:50. It contains an index, timestamp and 15 sensor channels. The supplied description states 15,169,480 instances and a 1 Hz sampling rate, but the actual CSV has 1,516,948 records with timestamps approximately 10 seconds apart. All project calculations use the actual CSV, and the discrepancy is documented rather than silently reconciled.

The selected continuous signals include TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current and Caudal_impulses. Four air-leak windows are reported in the accompanying dataset description. These are event-level intervals, not row-by-row confirmed fault labels.

## 4. Data preparation
The original source file is preserved. Derived 1-minute and 5-minute mean aggregates are provided. Empty time bins remain missing. The backend restores a regular time grid when loading the aggregates. For forecasting, it interpolates only isolated gaps of at most two five-minute bins and fits on the final contiguous usable segment; it does not compress long missing periods into adjacent time steps. For anomaly screening, the regular index is retained rather than dropping missing timestamps first.

## 5. Forecasting models
The application provides:

- **Holt trend:** level and additive trend exponential smoothing.
- **Holt-Winters:** additive trend and additive daily seasonality (288 five-minute samples per day); it requires at least four daily cycles.
- **Seasonal naive:** repeats the corresponding value from the previous daily cycle.
- **Hybrid:** Holt trend forecast plus the median of recent fitted residuals. This is a transparent statistical hybrid, not a deep-learning ensemble.
- **Prophet:** trend/changepoint model with daily and weekly seasonal components. Prophet is included as an optional dependency in `requirements-prophet.txt` and installed by the Windows launcher. If unavailable, the API marks it unavailable with an explicit reason.

These methods are appropriate lightweight candidates for recurring trends and seasonality. They do not fully model nonlinear vibration dynamics; the supplied dataset does not provide a dedicated vibration channel for this project. A future extension could add engineered nonlinear features or a compact tree-based residual model if suitable vibration data become available.

## 6. Evaluation methodology
Model comparison uses a chronological 80/20 holdout over the most recent 14 calendar days. No random shuffling is used. Forecasts are generated across the regular test horizon; metrics are calculated only where actual target values exist. The same `fit_predict` model implementation is used for both forecast output and holdout scoring, including the hybrid residual correction. MAE and RMSE are reported in native sensor units. MAPE excludes near-zero actual values because percentage error is unstable near zero. A model that cannot fit the selected sensor/window is shown as unavailable with a reason.

## 7. Anomaly detection and reported-event context
A rolling median estimates the local baseline. Absolute deviation is scaled by 1.4826 times the rolling median absolute deviation. A robust score above 4.5 is flagged, and nearby flagged observations are grouped into events. The backend also reports whether at least one flag occurs within each of the four source-reported air-leak windows.

This event overlap is a coarse contextual check only. It is not precision, recall, F1 or fault-classification accuracy because the source does not provide row-level fault labels and a flag inside a broad event window does not establish exact fault timing.

## 8. Limited-history adaptation experiment
The `/api/adaptation` endpoint takes a 10–15 day target window, compares a local-only Holt model with a Holt model initialized using level/trend estimates from earlier history, and evaluates both on a chronological holdout. The transferred level is re-anchored to the target window’s early observations to avoid blindly copying an earlier absolute sensor level.

This is a temporal warm-start proxy on one compressor time series. It is useful for illustrating how historical parameters might initialize a short-history model, but it is not evidence of cross-machine transfer learning. Validating transfer across machines requires machine identifiers and multiple machine histories, which are absent here.

## 9. Edge inference and connectivity
The Flask service binds to `127.0.0.1` and performs forecasting and analysis locally using local data and installed packages. Once installed, the dashboard does not depend on cloud inference or an active internet connection. This demonstrates the software architecture for offline edge inference. It does not prove production-grade latency or fit on a specific industrial gateway; those claims require target-hardware benchmarking for response time, memory and power.

## 10. Drift monitoring and safe model updates
The drift endpoint compares a historical reference window with a recent window using standardized mean shift and Population Stability Index (PSI). Thresholds are review triggers, not automatic fault decisions. The shadow-update endpoint compares Holt trend, Hybrid and Seasonal Naive candidates on a chronological holdout. A candidate is marked eligible for shadow promotion only if it improves MAE by at least 2% over the incumbent; the endpoint never replaces a deployed model.

A production system should additionally collect actual post-forecast readings, calculate rolling prediction errors, monitor missing-data rate and operating context, and trigger a review after sustained degradation. Candidate models should run in shadow mode, pass time-based validation and engineering approval, then be deployed with rollback capability. This prototype provides a recommendation mechanism, not a persistent model registry or production deployment controller.

## 11. Computational constraints
Holt, Holt-Winters, seasonal-naive and the residual-correction hybrid are relatively lightweight statistical methods. Prophet is more computationally expensive and is optional. A real industrial deployment should benchmark each candidate on the target gateway and select the smallest model that meets the required accuracy and latency. Local execution demonstrates offline behaviour but does not constitute hardware certification.

## 12. Requirement coverage summary

| Case-study challenge | Project response | Evidence / limitation |
|---|---|---|
| Historical data vs 10–15 day history | Temporal warm-start adaptation proxy | Same-series proxy only; no fleet transfer validation |
| Prediction latency | Local Flask inference | No target gateway benchmark |
| Unreliable connectivity | Offline local service | Requires initial package installation and local data |
| Recurring and irregular signals | Holt, Holt-Winters, Prophet, hybrid and robust anomaly screening | No dedicated vibration channel; nonlinear fault dynamics remain limited |
| Changing operating conditions | Mean-shift and PSI drift screen | Review signal only; not automatic adaptation |
| Limited compute | Lightweight statistical candidates | Prophet optional; gateway resource profiling remains future work |
| Safe model updates | Shadow candidate comparison and approval gate | No automatic deployment or persistent model registry |

## 13. Limitations
- One compressor/APU, not 120 independent machines across five factories.
- No dedicated vibration or operating-speed channel is used in this implementation.
- Four broad air-leak event windows, not row-level failure labels.
- No calibrated failure probability or remaining-useful-life model.
- Edge inference is demonstrated locally but not benchmarked on industrial hardware.
- Drift detection is a sensor-distribution screen; production prediction-error monitoring requires an ongoing stream of actual values.

## 14. Conclusion
SmartPredict provides an integrated local prototype for sensor forecasting, model comparison, anomaly screening, limited-history initialization, drift review and shadow model governance. It uses the complete supplied MetroPT-3 CSV and preserves source limitations. It addresses the main architectural themes of the case study while avoiding unsupported claims of fleet-wide transfer learning or confirmed failure prediction.

## References
- MetroPT-3, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
- Davari et al. (2021), *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry*, IEEE DSAA.

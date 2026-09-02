# Learned short-lead forecast correction: experiment record

Written 2026-09-02. Summary of the model development tracked in #7. The
executed notebooks in `notebooks/` are the full record; this document is
the condensed version. The forecast/observation archive itself is held
in a private database and is not committed: `notebooks/artifacts/` is
gitignored, so the notebooks reproduce from a local snapshot rather than
from data in this repository. Reported figures are aggregate statistics.

## 1. Problem

The application displays raw GFS 2 m temperature forecasts per station,
with a same-day empirical correction (`src/heat/bias.py`) that requires
at least three same-day forecast/observation pairs before it can act.
Objective: a trained correction with higher skill and full coverage,
using only information available at forecast time.

## 2. Data

Forecast/observation pairs accumulated by the 6-hourly refresh workflow
(`scripts/log_forecast_obs.py`) since 2026-07-15: 127,569 rows at the
time of the experiment, 164 stations, three metrics, of which the t2m
subset (42,533 station-time rows after pivoting) is used here. All pairs
have forecast lead at or below 8 h, a consequence of the logging
schedule; results therefore apply to short-lead correction only.

## 3. Evaluation protocol

- Temporal split V1: contiguous blocks of 33 / 7 / 8 days for train /
  validation / test on `forecast_valid_time`. Contiguous end-of-record
  blocks are used because forecast errors are strongly autocorrelated
  day to day, and because fit-on-history, deploy-forward matches the
  intended use.
- Model selection (hyperparameters, feature set) used the validation
  block only. The test block was evaluated once.
- All baselines are causal: each prediction uses only information
  available before the predicted instant.
- Spatial generalization: 35 of 164 stations withheld by a
  deterministic hash of the station id, never used in fitting for the
  generalization experiment.

## 4. Methods

Stage 1, pointwise regression of the forecast error. Features (10):
forecast t2m, forecast dewpoint depression, latitude, longitude,
station-local hour and day-of-year as sine/cosine harmonics, lead hours,
and the station mean error over the trailing 7 completed local days
(shifted one day so a row never uses its own day). Candidates: ridge
regression on standardized features, gradient-boosted trees (XGBoost,
early stopping on validation), and a multilayer perceptron (64/32,
early stopping).

Stage 2, spatial interpolation of the stage-1 residual field. An
exponential correlogram is fitted to train-block residuals (length
scale, nugget), a temporal decay constant is fitted to lagged residual
autocorrelation, and simple kriging maps each time slice of station
residuals to all station locations. Each forecast row uses the latest
slice at or before its own GFS init time, damped by the fitted temporal
factor. This is optimal interpolation with a static covariance, applied
at station locations; it corresponds to tier 1 of `docs/da/DESIGN.md`.

## 5. Results (test block, n = 12,881)

| configuration | test RMSE (°C) |
|---|---|
| raw GFS | 2.255 |
| global offset | 2.286 |
| same-day scheme (causal replica of production) | 2.151 |
| per-station offset | 2.031 |
| ridge, harmonic features | 1.785 |
| XGBoost | 1.612 |
| XGBoost + kriged residual | 1.469 |

Supporting results:

- The MLP reaches 2.422, below raw GFS, consistent with the training
  sample size; it is retained in the comparison as evidence for the
  model-class choice.
- The global offset is below raw GFS because the CONUS-wide mean error
  changed sign between the train and test periods.
- The same-day scheme acts on 66% of test rows (it requires three
  same-day pairs); all trained models act on 100%.
- At the 35 withheld stations: XGBoost 1.850 versus 2.375 raw; kriging
  of raw neighbor errors alone, with no per-station history, 2.270.
- Kriging applied to the per-station offset instead of stage 1 reaches
  1.842, indicating the two stages remove distinct error components.
- Feature importance (gain): local-hour harmonics 35%, trailing station
  error 24%, dewpoint depression 10%, remainder distributed.

## 6. Known limitations

1. The 95% quantile interval (XGBoost quantile regression) achieves
   79.7% empirical coverage on test. It is not suitable for display.
   Conformal calibration on the validation block is the planned fix and
   gates any deployment.
2. Applicability is limited to leads at or below 8 h and to warm-season
   conditions; the archive spans a single summer.
3. The stage-2 covariance is stationary, isotropic, and separable in
   space and time. These are the standard static-covariance limitations;
   the flow-dependent alternative is the EnSRF tier of the DA design.

## 7. Follow-up

In order: interval calibration; an inference module in `src/heat/` with
tests enforcing the causality rules above; shadow logging of corrections
in the refresh workflow, scored against arriving observations before any
user-facing change. Trained model files are committed when the
deployment PR needs them.

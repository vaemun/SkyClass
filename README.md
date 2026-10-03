# SkyClass

SkyClass predicts SDSS spectroscopic `STAR`, `GALAXY`, or `QSO` labels using five-band photometry, then reports how performance changes with sky position, brightness, measurement error, and QSO redshift. Redshift is diagnostic only and is never supplied to a classifier.

## Run

Use Python 3.11 or newer, install the packages, and train from the DR17 SkyServer SQL REST endpoint:

```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

The default pull is capped at 33,333 objects per class (up to 100,000 total). SkyServer's SQL REST parser rejects `UNION ALL`, so SkyClass makes one class-filtered query for each class and combines the results locally. Each query uses deterministic pseudorandom hash ordering: `ORDER BY CHECKSUM(s.specobjid), s.specobjid`. With a fixed DR17 catalogue and limit, this selects the same rows every time without selecting one contiguous sky patch. The expanded exact queries are stored in `data/raw_sdss_dr17_hash.query.json`.

SkyClass deliberately uses a capped-per-class sample rather than natural class proportions so all three labels have evaluation support; raw and retained counts are recorded. The earlier `data/raw_sdss_dr17.csv` cache is preserved unchanged and superseded. The new pull is cached separately at `data/raw_sdss_dr17_hash.csv`; reruns reuse it. Pass `--refresh` to fetch again, `--limit-per-class N` to change the cap, or `--output-dir PATH` to move generated model/report files. Duplicate `objid` rows are removed and conflicting labels for an object are excluded before splitting. The RA/Dec sample plot is generated at `artifacts/sky_coverage.png`.

Run the automated feature/leakage tests with:

```bash
python -m unittest discover -s tests -v
```

## Feature sets

All magnitudes are dereddened bandwise as `m_corrected = m - extinction`. The eight adjacent and non-adjacent colours are `u-g`, `g-r`, `r-i`, `i-z`, `u-r`, `g-i`, `r-z`, and `u-z`.

| Set | Inputs |
|---|---|
| A | Dereddened PSF colours only |
| B | A plus dereddened PSF r magnitude |
| C | A plus per-band `psfMag - modelMag` concentration |
| D | C plus propagated PSF colour uncertainties |

Colour uncertainty is calculated as `sqrt(err_a^2 + err_b^2)`. PSF and model photometry are not mixed for colours: every colour set uses PSF magnitudes; model magnitudes only form the explicit concentration features. RA/Dec define 15-degree RA by 10-degree Dec spatial groups and are not model features. Object IDs are retained only in the raw cache; redshift is retained only for the post-hoc QSO diagnostic.

## Validation and error analysis

The report compares five-fold `GroupKFold` over 15-degree RA by 10-degree Dec blocks with shuffled five-fold stratification, using the same four models, PSF-colour features, and fixed settings. Fold means and standard deviations are reported. An additional grouped spatial holdout is retained for the feature ablation and detailed error slices. Hyperparameters for that holdout are selected from three modest XGBoost candidates on a separate grouped validation split within its training partition.

The bright-to-faint experiment fits a single feature-D XGBoost model using only `r < 18` rows and evaluates on `r > 19`. It reports class recall and macro-F1 with class-stratified bootstrap intervals. Its control evaluates the same bright-trained model on a bright holdout and on a faint holdout sampled to the exact same size and class counts. Error analyses count rows with any PSF-band magnitude error >= 1 mag by class and dereddened-r bin, then recompute spatial-test uncertainty quintiles both including and excluding those rows. QSO recall is tabulated across spectroscopic redshift bins, strictly as a diagnostic.

The labelled sample is spectroscopically targeted using selection rules that depend on observed properties including colour and brightness. Consequently, scores quantify this capped spectroscopic sample and do not estimate performance on every photometric SDSS detection. Spatial blocking reduces neighbour leakage; it does not remove targeting bias.

Additional diagnostics include Gaussian Monte Carlo photometric-noise augmentation, an S/N-weighted model, multiclass Brier score and top-label ECE with reliability bins, accuracy-versus-coverage under abstention, row-bootstrap 95% intervals, XGBoost TreeSHAP and permutation importance, a majority-class baseline, split class proportions, and a train/test `objid`-overlap assertion. Noise augmentation assumes independent Gaussian magnitude errors and should be interpreted cautiously for extreme reported uncertainties.

Phase 3 adds one-vs-rest isotonic calibration fitted on a spatially grouped calibration fold drawn only from the spatial training partition; raw and calibrated probabilities are compared on the untouched spatial test fold overall and by dereddened `r` bin. Abstention is evaluated using both maximum calibrated class probability and the RMS signal-to-noise across the eight PSF colours. Baseline, Monte Carlo-augmented, and S/N-weighted models are compared in faint `r` bins with per-class stratified bootstrap intervals and paired macro-F1 intervals. SHAP and permutation importance are reported separately for feature sets A and D. Generated figures are under `artifacts/figures/`, with 2–4 sentence interpretations in the generated report.

## Results

Run-specific findings are generated from the current cached sample. Read `artifacts/REPORT.md` for the concise report and `artifacts/metrics.json` for per-class metrics, confusion matrices, redshift bins, calibration, abstention, uncertainty analyses, feature importance, confidence intervals, split checks, and cleaning counts.
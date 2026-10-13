# AI-Based Air Quality Awareness Assistant — MLOps Milestone 1: Implementation Plan (AI-Executable Prompt)

> **How to use this document:** Paste the whole file into an AI coding assistant (Claude Code, Cursor, etc.) as the task prompt. Ask it to execute **one phase at a time**, show the files it creates, and wait for confirmation before moving on. Items marked **[HUMAN]** cannot be done by an AI (approvals, accounts, screenshots) and must be done by the students.

---

## 0. ROLE AND GROUND RULES FOR THE AI

You are a senior MLOps engineer helping two undergraduate students (course AI252TA, Machine Learning Operations) build the first milestone of the project **"AI-Based Air Quality Awareness Assistant"**.

Ground rules:
1. Build everything in a **single reproducible Git repository** with a clean structure (Section 3).
2. Every script must be runnable from the command line, read its settings from `params.yaml`, and be wired into a DVC pipeline (`dvc.yaml`) so that `dvc repro` rebuilds everything end to end.
3. **No notebook-only logic.** Notebooks are allowed for EDA only; any logic reused later must live in `src/`.
4. **No data leakage.** All splits are chronological. Scalers/imputers are fit on the training split only.
5. Set a global random seed (`seed: 42`) in `params.yaml` and use it everywhere.
6. Never commit raw data or model binaries to Git. Git tracks code, configs, `.dvc` files, and `dvc.lock`; DVC tracks data and artifacts.
7. Use Python 3.10+, type hints, docstrings, `logging` (not `print`), and `pathlib`.
8. After each phase, print a short "what was done / how to verify" summary and the exact commands to run.
9. Produce **publication-quality figures** (labelled axes, units, titles, 300 dpi PNG) saved to `reports/figures/`.
10. If a design decision is ambiguous, pick the simpler option, state the assumption in `docs/decisions.md`, and continue.

---

## 1. PROJECT CONTEXT AND SCOPE

**Project title:** AI-Based Air Quality Awareness Assistant
**Long-term vision:** A system that predicts air quality and turns it into plain-language health guidance (e.g., "AQI will be Poor tomorrow; sensitive groups should limit outdoor activity").
**This milestone covers only the ML foundation + MLOps tooling:**

| # | Deliverable |
|---|---|
| 1 | Dataset acquisition and approval |
| 2 | Data exploration (EDA) |
| 3 | Feature engineering |
| 4 | Baseline models |
| 5 | Git repository |
| 6 | DVC dataset versioning |
| 7 | MLflow experiment tracking |
| 8 | Initial model evaluation |

**Out of scope for this milestone** (do not build now, but design so they can be added later): REST API (FastAPI), Docker, CI/CD, monitoring/drift detection, LLM-based advice chatbot, deployment.

**ML problem definition (two linked tasks):**
- **Task A — Regression:** Predict **next-day AQI** (numeric, 0–500) for a city from today's and recent pollutant readings.
- **Task B — Classification:** Predict **next-day AQI category** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) per the Indian CPCB scale. Derive it from the Task A target so both stay consistent.

**Team:** 2 students (note: course circular standard is 4; this is an approved/declared 2-person group).

---

## 2. TECH STACK

| Concern | Tool |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy, pyarrow |
| Viz | matplotlib, seaborn, (optional) plotly |
| ML | scikit-learn, xgboost, lightgbm |
| Versioning | Git + GitHub, DVC |
| Experiment tracking | MLflow (local file or SQLite backend) |
| Config | PyYAML (`params.yaml`) |
| Quality | pytest, ruff (or flake8), black, pre-commit |
| Environment | `venv` + `requirements.txt` (pin versions) |

---

## 3. TARGET REPOSITORY STRUCTURE

Create exactly this layout:

```
aqi-awareness-assistant/
├── .dvc/                      # created by dvc init
├── .github/                   # (empty for now; CI comes later)
├── data/
│   ├── raw/                   # DVC-tracked, never edited
│   ├── interim/               # cleaned data (DVC output)
│   └── processed/             # model-ready features (DVC output)
├── docs/
│   ├── dataset_approval.md    # approval proposal + faculty response
│   ├── data_card.md           # dataset documentation
│   └── decisions.md           # design decisions log
├── models/                    # saved baseline models (DVC output)
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   ├── figures/
│   ├── metrics/               # JSON metrics files (DVC metrics)
│   └── milestone1_report.md
├── src/
│   ├── __init__.py
│   ├── config.py              # loads params.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py          # validate + copy raw data
│   │   ├── clean.py           # cleaning + imputation
│   │   └── validate.py        # schema + sanity checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── aqi_utils.py       # AQI bucket mapping
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py       # persistence, seasonal naive, etc.
│   │   ├── train.py           # trains all baselines, logs to MLflow
│   │   └── evaluate.py        # metrics, plots
│   └── utils/
│       ├── __init__.py
│       └── logging_utils.py
├── tests/
│   ├── test_aqi_utils.py
│   ├── test_features.py
│   └── test_no_leakage.py
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── dvc.yaml
├── params.yaml
├── pyproject.toml             # ruff/black config
├── requirements.txt
└── README.md
```

---

## 4. PHASE 1 — DATASET ACQUISITION AND APPROVAL

### 4.1 Candidate datasets (evaluate all, recommend one)

| Candidate | Description | Pros | Cons |
|---|---|---|---|
| **A (recommended primary): CPCB India city-day air quality, 2015–2020** (commonly distributed on Kaggle as "Air Quality Data in India", file `city_day.csv`) | Daily pollutant readings (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene) + `AQI` + `AQI_Bucket` for ~26 Indian cities including Bengaluru | Matches Indian AQI scale; includes target; ~29k rows; small enough for laptops; many missing values give real cleaning work | Ends in 2020 (COVID-era distortion in 2020); missingness is heavy |
| B (optional enrichment): OpenAQ API data for Bengaluru stations | Recent measurements from reference/low-cost stations | Fresh data | Needs API handling, inconsistent coverage |
| C (optional enrichment): Weather data (temperature, humidity, wind) from Open-Meteo / NASA POWER | Meteorological drivers of pollution | Improves forecasts | Extra join logic and time-zone issues |

**Decision for this milestone:** Use **Candidate A as the only required dataset**. Keep B and C as documented "future enrichment" in `docs/decisions.md`.

### 4.2 Tasks

1. **[HUMAN]** Download the dataset file(s) from the source. Record exact URL, download date, file name, file size, and SHA-256 hash. Verify the license/terms of use on the source page and record them (do not assume).
2. Place the file at `data/raw/city_day.csv` (do not modify it).
3. Generate `docs/data_card.md` with these sections:
   - Source and URL, license, download date, SHA-256
   - Collection method and provenance (CPCB monitoring stations; state what is known and what is unknown)
   - Row/column counts, time range, cities covered
   - Column dictionary (name, type, unit, description, % missing)
   - Known limitations (missingness, station changes, COVID-19 lockdown period, no weather variables, city-level aggregation)
   - Ethical/privacy note (no personal data; public environmental data)
   - Intended use and non-intended use (not for medical decisions)
4. Generate `docs/dataset_approval.md` as a **one-page approval request** to the course faculty with: project title, problem statement, dataset name/source/license, size, target variable, why it suits the task, risks, and the planned train/validation/test strategy. Leave a signed-off/approval section at the bottom.
5. **[HUMAN]** Submit the approval request to the faculty and record the response (date, name, decision, comments) in the same file. Take a screenshot/email copy for the report.
6. Write `src/data/validate.py` that asserts:
   - Required columns exist: `City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket`
   - `Date` parses to datetime; no future dates
   - No duplicate `(City, Date)` pairs (report and handle if found)
   - Numeric pollutant columns are non-negative
   - Reports rows per city and date range per city
   - Writes a validation summary JSON to `reports/metrics/data_validation.json`
7. Write `src/data/ingest.py` that loads the raw CSV, runs validation, and exits non-zero on hard failures.

**Acceptance criteria:** `data/raw/city_day.csv` exists; `data_card.md` and `dataset_approval.md` are complete; `python -m src.data.validate` passes and writes the JSON.

---

## 5. PHASE 2 — GIT REPOSITORY

1. **[HUMAN]** Create an empty GitHub repo named `aqi-awareness-assistant` (private until submission, or as faculty requires). Add the teammate as a collaborator.
2. Locally: `git init`, set default branch to `main`, add remote.
3. Create `.gitignore` covering: `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `mlruns/`, `mlartifacts/`, `*.db` (MLflow sqlite), `.env`, `data/raw/*` (but keep DVC `.dvc` files), `data/interim/*`, `data/processed/*`, `models/*`, `.DS_Store`. (DVC adds entries automatically; make sure nothing conflicts.)
4. Create `README.md` skeleton (project title, problem, quick start, repo structure, how to reproduce, team, license). Finalize in Phase 9.
5. Create `requirements.txt` with **pinned** versions (install, then `pip freeze` the relevant packages): pandas, numpy, pyarrow, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, mlflow, dvc, pyyaml, pytest, ruff, black, pre-commit, jupyter.
6. Set up `pre-commit` with ruff + black + trailing-whitespace + end-of-file-fixer + check-added-large-files (max 500 KB).
7. **Branching strategy** (document in README):
   - `main`: always runnable, protected by convention (merge via PR only)
   - `feature/<short-name>` branches, one per phase (e.g., `feature/dvc-setup`, `feature/eda`, `feature/features`, `feature/baselines`, `feature/mlflow`)
   - Each teammate reviews the other's pull request (evidence of collaboration for grading)
8. **Commit message convention** (Conventional Commits): `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`. Make **small, frequent commits**; aim for a visible, meaningful history (20+ commits across both authors).
9. Tag the end of this milestone: `git tag -a v0.1-milestone1 -m "Milestone 1"`.

**Acceptance criteria:** Repo pushed to GitHub with both contributors' commits, at least 4 merged PRs, pre-commit hooks working, tagged release.

---

## 6. PHASE 3 — DVC DATASET VERSIONING

### 6.1 Setup
1. `dvc init` and commit `.dvc/` + `.dvcignore`.
2. Configure a **remote**. Priority order (choose the first that works for the students):
   - (a) A local/shared folder remote (`dvc remote add -d localremote ../dvc-storage`) — simplest, always works
   - (b) DagsHub or Google Drive remote — better for demonstrating "real" remote storage
   Document the choice in `docs/decisions.md`. Never commit credentials; use `dvc remote modify --local`.
3. Track the raw data: `dvc add data/raw/city_day.csv`; commit the generated `data/raw/city_day.csv.dvc` and updated `.gitignore`; run `dvc push`.

### 6.2 Demonstrate versioning (this is what graders look for)
1. **Version v1:** original raw data → commit + tag `data-v1`.
2. **Version v2:** create a deliberate, documented change, e.g. produce `data/raw/city_day.csv` filtered/appended variant (for example, add a second reference file such as `data/raw/stations_meta.csv`, or replace the file with a version where you drop the clearly corrupt rows identified in EDA). Run `dvc add`, commit, tag `data-v2`, `dvc push`.
3. Show switching versions: `git checkout data-v1 && dvc checkout` then back to `data-v2`. Capture terminal output/screenshots for the report.
4. Show `dvc diff data-v1 data-v2` output.

### 6.3 DVC pipeline (`dvc.yaml`)
Define these stages (each with `cmd`, `deps`, `params`, `outs`/`metrics`):

```yaml
stages:
  validate:
    cmd: python -m src.data.ingest
    deps: [src/data/ingest.py, src/data/validate.py, data/raw/city_day.csv]
    metrics:
      - reports/metrics/data_validation.json:
          cache: false

  clean:
    cmd: python -m src.data.clean
    deps: [src/data/clean.py, data/raw/city_day.csv]
    params: [clean]
    outs: [data/interim/city_day_clean.parquet]

  features:
    cmd: python -m src.features.build_features
    deps:
      - src/features/build_features.py
      - src/features/aqi_utils.py
      - data/interim/city_day_clean.parquet
    params: [features, split, seed]
    outs:
      - data/processed/train.parquet
      - data/processed/val.parquet
      - data/processed/test.parquet

  train:
    cmd: python -m src.models.train
    deps:
      - src/models/train.py
      - src/models/baselines.py
      - data/processed/train.parquet
      - data/processed/val.parquet
    params: [train, seed]
    outs: [models/]
    metrics:
      - reports/metrics/val_metrics.json:
          cache: false

  evaluate:
    cmd: python -m src.models.evaluate
    deps:
      - src/models/evaluate.py
      - models/
      - data/processed/test.parquet
    outs: [reports/figures/evaluation/]
    metrics:
      - reports/metrics/test_metrics.json:
          cache: false
```

Run `dvc repro`, confirm `dvc dag` renders, commit `dvc.yaml` + `dvc.lock`. Demonstrate that changing one param in `params.yaml` only re-runs affected stages.

### 6.4 `params.yaml` (create with these keys)

```yaml
seed: 42
data:
  raw_path: data/raw/city_day.csv
clean:
  max_missing_frac_row: 0.6      # drop rows with > 60% pollutant values missing
  min_rows_per_city: 700         # drop cities with too little history
  impute_method: ffill_limit     # ffill within city, limited gap
  ffill_limit_days: 3
  drop_cols_over_missing: 0.7    # drop columns missing > 70% overall (e.g., Xylene) -- decided after EDA
features:
  target_col: AQI
  horizon_days: 1
  lags: [1, 2, 3, 7]
  rolling_windows: [3, 7, 14]
  use_calendar: true
  use_city_onehot: true
split:
  train_end: "2018-12-31"
  val_end: "2019-12-31"          # test = 2020-01-01 onward (note COVID effect; see decisions.md)
train:
  models: [persistence, seasonal_naive, ridge, random_forest, xgboost, lightgbm]
  cv_folds: 4                    # TimeSeriesSplit on train
```

**Acceptance criteria:** `dvc remote list` shows a remote; two data versions tagged; `dvc repro` runs clean end to end; `dvc dag` output captured.

---

## 7. PHASE 4 — DATA EXPLORATION (EDA)

Create `notebooks/01_eda.ipynb` (and export key figures to `reports/figures/eda/`). Use **raw** data for diagnosing issues, then confirm cleaning decisions. Each section must end with a one-sentence **"Finding → Decision"** markdown cell that gets copied into `docs/decisions.md`.

### 7.1 Required analyses
1. **Structure:** shape, dtypes, memory, head/tail, date range, number of cities, rows per city (bar chart).
2. **Missingness:**
   - % missing per column (bar chart)
   - Missingness heatmap (city × pollutant, using fraction missing)
   - Missingness over time per city (identify stations that start/stop reporting)
   - Fraction of rows where `AQI` itself is missing (cannot be used as target; decide to drop)
3. **Distributions:**
   - Histograms/KDE of each pollutant (log-scale where skewed)
   - Boxplots per pollutant to identify outliers; list values violating physical plausibility (negatives, absurdly large)
   - Class distribution of `AQI_Bucket` (overall and per city) — quantify imbalance
4. **Temporal patterns:**
   - Time series of AQI for 4–5 major cities (Delhi, Bengaluru, Mumbai, Kolkata, Chennai)
   - Monthly seasonality (month × city heatmap of mean AQI; winter vs monsoon contrast)
   - Day-of-week effect (boxplot)
   - Annual trend; clear annotation of **COVID-19 lockdown period (Mar–Jun 2020)**
5. **Relationships:**
   - Pearson **and** Spearman correlation heatmaps among pollutants and AQI
   - Scatter of PM2.5 vs AQI and PM10 vs AQI (expect PM2.5/PM10 to dominate AQI)
   - Autocorrelation (ACF) and partial autocorrelation (PACF) of AQI for one city → justifies lag features
6. **City comparison:** mean/median/90th percentile AQI per city (sorted bar chart); highlight Bengaluru as the local focus.
7. **Target sanity check:** verify `AQI_Bucket` is consistent with `AQI` per CPCB thresholds; report inconsistencies.
8. **Baseline difficulty check:** compute the day-over-day AQI change distribution (how hard is next-day prediction? A persistence baseline sets the bar).

### 7.2 EDA outputs
- ≥ 12 clean figures in `reports/figures/eda/`
- `docs/eda_summary.md` with a bullet list of the 8–10 most important findings and the resulting decisions (e.g., "Xylene is 60%+ missing → drop", "2020 shows lockdown anomaly → use as test and discuss", "Severe class is rare → use macro-F1").

**Acceptance criteria:** Notebook runs top-to-bottom with `Restart & Run All`; figures saved; summary written.

---

## 8. PHASE 5 — FEATURE ENGINEERING

Implement in `src/data/clean.py`, `src/features/aqi_utils.py`, and `src/features/build_features.py`.

### 8.1 Cleaning (`clean.py`)
1. Parse `Date`; sort by `(City, Date)`; drop exact duplicates.
2. Drop rows with missing `AQI` (target).
3. Drop cities with fewer than `min_rows_per_city` rows.
4. Replace physically impossible values (negative concentrations) with NaN.
5. Winsorize extreme outliers per pollutant at the 0.5th/99.5th percentile **computed on the training period only** (pass the training cutoff date; do not leak test statistics).
6. Drop columns exceeding `drop_cols_over_missing`.
7. **Imputation:** within each city, forward-fill up to `ffill_limit_days`; remaining gaps filled with the **city's training-period median** for that pollutant; add boolean `<col>_was_missing` indicator columns for the main pollutants (PM2.5, PM10, NO2, O3, CO, SO2). Document why.
8. Save to `data/interim/city_day_clean.parquet`.

### 8.2 AQI utilities (`aqi_utils.py`)
Implement `aqi_to_bucket(aqi: float) -> str` using the CPCB scale:

| AQI | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

Also `bucket_to_int` / `int_to_bucket`, and a `HEALTH_ADVICE` dict (one short sentence per category) that the future "assistant" will use — include it now, unit-tested, but do not build the chatbot. Write `tests/test_aqi_utils.py` covering boundaries (50, 51, 100, 101, 200, 201, 300, 301, 400, 401, 500) and invalid inputs.

### 8.3 Feature construction (`build_features.py`)
All features computed **per city, using only information available at day *t*** to predict day *t+1*.

1. **Target creation:** `target_aqi = AQI.shift(-1)` within city; `target_bucket = aqi_to_bucket(target_aqi)`. Drop the last row of each city (no target).
2. **Lag features:** for `AQI` and key pollutants (PM2.5, PM10, NO2, O3, CO, SO2): lags `[1, 2, 3, 7]` days (`AQI_lag1`, …). Lag 0 is the current day's value.
3. **Rolling statistics** (shifted by 1 so the window never includes the target day): mean, std, and max over `[3, 7, 14]` days for AQI and PM2.5.
4. **Change features:** `AQI_diff1 = AQI_t − AQI_{t-1}`, `PM25_diff1`.
5. **Calendar features:** day-of-week, month, day-of-year, `is_weekend`, and **cyclical encodings** (`sin/cos` of month and day-of-year). Season flag (winter: Nov–Feb, summer: Mar–May, monsoon: Jun–Sep, post-monsoon: Oct).
6. **Ratio features:** `PM25_PM10_ratio` (guard divide-by-zero), `NO2_O3_ratio` (optional).
7. **City encoding:** one-hot encode `City` (for linear models) and keep an integer code (for tree models); make this a param.
8. **Chronological split** using `params.yaml`: train ≤ 2018-12-31, validation 2019, test 2020-01-01 onward. Assert no date overlap between splits (`tests/test_no_leakage.py`). Print split sizes and class balance for each.
9. **Scaling:** `StandardScaler` fit on train only, applied to val/test; save the fitted scaler (joblib) to `models/scaler.joblib`. Tree models receive unscaled features; linear models receive scaled.
10. Save `train.parquet`, `val.parquet`, `test.parquet` to `data/processed/` plus `data/processed/feature_list.json` (ordered feature names).
11. **Feature importance preview:** quick mutual-information or correlation ranking of features vs `target_aqi` on train only → `reports/figures/features/feature_ranking.png`.

### 8.4 Tests (`tests/test_features.py`, `tests/test_no_leakage.py`)
- Lags at row *t* equal the original value at *t−k* within the same city (and never cross city boundaries).
- Rolling features at row *t* do not use the value at *t+1* (construct a synthetic series and assert).
- Max train date < min val date < min test date.
- No NaN remains in the processed feature matrices.
- `target_bucket` matches `aqi_to_bucket(target_aqi)`.

**Acceptance criteria:** `dvc repro features` creates three parquet files; all tests pass with `pytest -q`.

---

## 9. PHASE 6 — BASELINE MODELS

Implement in `src/models/baselines.py` and `src/models/train.py`.

### 9.1 Naive baselines (no training; set the floor)
1. **Persistence:** predict tomorrow's AQI = today's AQI.
2. **Seasonal naive (weekly):** predict tomorrow's AQI = AQI from 6 days ago (same weekday as t+1, i.e., `AQI_lag6`; compute lag 6 explicitly if not in the lag list) — or use the same-day-last-week value; state the exact definition in code docstring.
3. **City-month mean:** predict the training-period mean AQI for that city and month.
4. **Majority class (classification):** always predict the most common bucket.

### 9.2 ML baselines — Regression (Task A)
- `Ridge` (alpha tuned over `[0.1, 1, 10, 100]` via TimeSeriesSplit on train)
- `RandomForestRegressor` (n_estimators=300, max_depth=None, min_samples_leaf=3, n_jobs=-1)
- `XGBRegressor` (n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, early stopping on validation)
- `LGBMRegressor` (similar settings)
Keep hyperparameter search **minimal** — this is a *baseline* milestone. A small grid or defaults are fine; heavy tuning belongs to a later milestone.

### 9.3 ML baselines — Classification (Task B)
- `LogisticRegression` (multinomial, `class_weight="balanced"`, scaled features)
- `RandomForestClassifier` (`class_weight="balanced_subsample"`)
- `XGBClassifier` or `LGBMClassifier` (use class weights/sample weights)
- Also evaluate the **"regress-then-bucket"** approach: take the best regression model's predicted AQI and convert it with `aqi_to_bucket` — compare it with direct classifiers.

### 9.4 Training script behavior (`train.py`)
1. Load `params.yaml`, set seeds.
2. Load train/val parquet and `feature_list.json`.
3. For each model in `params.train.models`: fit on train, predict on val, compute metrics (Section 11), log everything to MLflow (Section 10), and save the fitted model to `models/<name>.joblib`.
4. Write a consolidated `reports/metrics/val_metrics.json` (per-model dict of metrics).
5. Print a leaderboard sorted by validation RMSE (regression) and macro-F1 (classification).
6. Use `TimeSeriesSplit` (not shuffled KFold) wherever CV is used.

**Acceptance criteria:** All listed baselines train in under ~10 minutes on a laptop; models saved; leaderboard printed; naive baselines clearly worse than or comparable to ML models (if an ML model does *not* beat persistence, report that honestly and investigate).

---

## 10. PHASE 7 — MLFLOW EXPERIMENT TRACKING

### 10.1 Setup
1. Use a local tracking store: `mlflow.set_tracking_uri("sqlite:///mlflow.db")` with artifact root `./mlartifacts` (both git-ignored; optionally DVC-track `mlartifacts`).
2. Experiments: `aqi-regression-baselines` and `aqi-classification-baselines`.
3. Launch UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`. **[HUMAN]** capture screenshots.

### 10.2 What to log per run
- **Run name:** `<model>_<short-timestamp>`
- **Tags:** `task` (regression/classification), `model_family` (naive/linear/tree/boosting), `git_commit` (from `git rev-parse HEAD`), `dvc_data_version` (the `md5` from `data/raw/city_day.csv.dvc` or the git tag), `author`, `stage=baseline`
- **Params:** every hyperparameter, `seed`, `horizon_days`, `lags`, `rolling_windows`, `train_end`, `val_end`, number of features, train/val sizes
- **Metrics:** val metrics (and CV metrics as `cv_rmse_mean`, `cv_rmse_std`); training time in seconds
- **Artifacts:** the model (`mlflow.sklearn.log_model` / `mlflow.xgboost.log_model` / `mlflow.lightgbm.log_model` with `signature` and `input_example`), `params.yaml`, `feature_list.json`, feature-importance plot, predicted-vs-actual scatter, residual plot, confusion matrix (classification), classification report text
- Use `mlflow.autolog()` **only** if you also log the custom items above; prefer explicit logging for clarity.

### 10.3 Experiment comparison (required for the report)
1. In the MLflow UI, select all regression runs → compare view → screenshot of metrics parallel-coordinates/bar chart.
2. Programmatic comparison script `src/models/compare_runs.py` using `mlflow.search_runs` → saves `reports/metrics/run_comparison.csv` sorted by validation RMSE.
3. Run **at least two parameter variations** for one model (e.g., XGBoost with `max_depth` 4 vs 6, or with/without rolling features) to demonstrate that tracking enables comparison. Log an ablation: **with vs without lag features** — this answers "do the engineered features help?".
4. Optionally register the best model in the MLflow Model Registry as `aqi-regressor` with stage/alias `candidate` (full registry/promotion workflow is a later milestone).

**Acceptance criteria:** ≥ 10 logged runs across two experiments; each has params/metrics/artifacts/tags; comparison CSV generated; screenshots taken.

---

## 11. PHASE 8 — INITIAL MODEL EVALUATION

Implement in `src/models/evaluate.py`; run the **test set exactly once** at the end, on the selected best model(s) plus all naive baselines for context.

### 11.1 Metrics
**Regression:** MAE, RMSE, R², MAPE (guard zero), median absolute error, and **"within ±20 AQI points" rate**.
**Classification:** accuracy, **macro-F1**, weighted-F1, per-class precision/recall/F1, **balanced accuracy**, Cohen's kappa, and **"within one category" accuracy** (a prediction one bucket off is far less harmful than five off). Also report **recall for Poor/Very Poor/Severe combined** — missing a dangerous day is the costly error for a health-awareness system.

### 11.2 Required analyses
1. Leaderboard table: all models × all metrics (validation and test) → `reports/metrics/test_metrics.json` and a Markdown table in the report.
2. Skill score vs persistence: `1 − RMSE_model / RMSE_persistence`.
3. Plots (to `reports/figures/evaluation/`): predicted vs actual scatter with y=x line; residual histogram; residuals over time; confusion matrix (normalized); per-class F1 bar chart.
4. **Error breakdown:** RMSE/MAE **by city**, **by season/month**, **by AQI level** (low/medium/high), and **by weekday vs weekend**. Identify where the model fails (typically sudden spikes and high-AQI days).
5. **Distribution shift check:** compare validation (2019) vs test (2020) performance; discuss the COVID lockdown effect honestly. Optionally report test metrics excluding Mar–Jun 2020.
6. **Feature importance:** permutation importance on validation (not just built-in importances) for the best tree model; top-15 bar chart.
7. **Bengaluru spotlight:** a dedicated mini-evaluation for Bengaluru (the local focus) with a time-series plot of predicted vs actual AQI.
8. **Limitations & next steps** list: no meteorology, city-level aggregation, short history, class imbalance, 1-day horizon only; next milestones (add weather features, hyperparameter tuning, multi-day horizon, API, Docker, CI/CD, monitoring, LLM advice layer).

**Acceptance criteria:** `test_metrics.json` written; all plots saved; evaluation numbers reproducible via `dvc repro evaluate`.

---

## 12. PHASE 9 — REPRODUCIBILITY, DOCUMENTATION, AND SUBMISSION PACKAGE

1. **README.md** must contain: project summary; problem definition; dataset + data card link; architecture/pipeline diagram (Mermaid: raw → validate → clean → features → train → evaluate, with Git/DVC/MLflow annotated); setup (`python -m venv`, `pip install -r requirements.txt`); `dvc pull`; `dvc repro`; how to launch MLflow UI; results table; repo structure; team contributions; limitations; license.
2. **Fresh-clone test (mandatory):** in a new directory run `git clone … && python -m venv … && pip install -r requirements.txt && dvc pull && dvc repro && pytest`. Fix anything that breaks. This is the single best proof of reproducibility.
3. **`reports/milestone1_report.md`** (convert to PDF/DOCX if the course requires it) with sections: (1) Introduction & problem; (2) Dataset & approval; (3) EDA key findings (with figures); (4) Feature engineering (table of features + leakage safeguards); (5) Baseline models & setup; (6) Git workflow (screenshot of commit graph, PRs); (7) DVC versioning (screenshots: `dvc dag`, `dvc diff`, version switch); (8) MLflow tracking (screenshots of experiment list and run comparison); (9) Initial evaluation (tables + plots + error analysis); (10) Limitations & future work; (11) Team contribution table; (12) References.
4. **Demo script** for a viva/demo (5 minutes): show repo → `dvc dag` → switch data version → `dvc repro` (only changed stages rerun) → MLflow UI comparison → test results → one error-analysis insight.
5. Tag `v0.1-milestone1`, push tags, `dvc push`.

---

## 13. TWO-PERSON WORK SPLIT (SUGGESTED)

| Area | Student A | Student B |
|---|---|---|
| Phase 1 dataset + approval | data card, validation script | approval doc, submission |
| Phase 2–3 Git + DVC | repo, branching, pre-commit | DVC remote, versions v1/v2, `dvc.yaml` |
| Phase 4 EDA | sections 7.1 (1–4) | sections 7.1 (5–8) |
| Phase 5 features | cleaning + AQI utils + tests | feature builder + split + leakage tests |
| Phase 6 baselines | naive + linear + RF | XGBoost + LightGBM + classifiers |
| Phase 7 MLflow | logging utilities, tags | comparison script, ablations |
| Phase 8 evaluation | metrics + plots | error breakdown + Bengaluru spotlight |
| Phase 9 docs | README, diagram | report, screenshots, fresh-clone test |

Each student **reviews** the other's PRs. Record contributions in the report.

---

## 14. SUGGESTED TIMELINE (≈ 2–3 weeks; compress as needed)

| Days | Work |
|---|---|
| 1–2 | Phase 1 (dataset, approval request sent) + Phase 2 (repo) |
| 3 | Phase 3 (DVC init, remote, v1) |
| 4–6 | Phase 4 (EDA) + data v2 decision |
| 7–9 | Phase 5 (cleaning, features, tests) |
| 10–12 | Phase 6 + Phase 7 (baselines with MLflow logging) |
| 13–14 | Phase 8 (evaluation) |
| 15–17 | Phase 9 (README, fresh-clone test, report, demo rehearsal) |

---

## 15. DEFINITION OF DONE (CHECKLIST)

- [ ] Dataset approved by faculty; `data_card.md` complete
- [ ] `data/raw` tracked by DVC with remote; tags `data-v1`, `data-v2`; version switch demonstrated
- [ ] `dvc repro` runs the full pipeline from a fresh clone
- [ ] EDA notebook with ≥ 12 figures and `eda_summary.md`
- [ ] Features documented, tested, and leakage-free; chronological split
- [ ] ≥ 4 naive/ML regression baselines and ≥ 3 classification baselines trained
- [ ] ≥ 10 MLflow runs with params, metrics, artifacts, tags; comparison CSV + screenshots
- [ ] Test-set evaluation with error breakdown, Bengaluru spotlight, COVID discussion
- [ ] `pytest` passes; ruff/black clean
- [ ] README, report, and demo script finished; repo tagged `v0.1-milestone1`

---

## 16. COMMON PITFALLS (THE AI MUST AVOID THESE)

1. **Random train/test split on time series** → leakage. Always split by date.
2. **Computing rolling/lag features after the split** or across city boundaries → leakage/corruption. Group by city; compute before splitting; verify with tests.
3. **Fitting imputers/scalers/winsorization limits on the full dataset** → leakage.
4. **Using the current day's `AQI_Bucket` as a feature to predict the next day's bucket without acknowledging it's essentially persistence**; always compare against the persistence baseline.
5. **Reporting accuracy only** on imbalanced classes → use macro-F1, balanced accuracy, per-class recall.
6. **Tuning on the test set.** Test is touched once, at the end.
7. **Committing data or `mlruns/` to Git.**
8. **Hard-coded paths and magic numbers** → everything goes through `params.yaml`.
9. **Non-reproducible runs** (unset seeds, unpinned dependencies).
10. **Overstating results**: if models only marginally beat persistence, say so; discuss why (strong day-to-day autocorrelation, missing meteorology).

---

## 17. FIRST INSTRUCTION TO THE AI

Start with **Phase 2 skeleton + Phase 1 scaffolding**: create the full repository structure from Section 3, `requirements.txt`, `.gitignore`, `params.yaml`, `pyproject.toml`, pre-commit config, empty module files with docstrings, and the `data_card.md` and `dataset_approval.md` templates. Then stop, list what you created, and ask me to confirm before proceeding to Phase 3 (DVC).

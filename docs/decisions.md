# Architecture & Design Decisions Log

This document records key engineering, data science, and design decisions made throughout the project, following the standard Architecture Decision Record (ADR) pattern.

---

## ADR-001: Primary Dataset Selection
- **Status:** Accepted
- **Context:** We evaluated three candidates: (A) CPCB India city-day data (2015–2020), (B) OpenAQ real-time station data, and (C) Meteorological data from Open-Meteo / NASA POWER.
- **Decision:** Use Candidate A (`city_day.csv`) as the primary and required dataset for Milestone 1. Candidate B and C are reserved for future enrichments.
- **Rationale:** Candidate A already includes officially computed CPCB AQI targets and categories, covers 26 cities across 5.5 years, and poses realistic data quality challenges (missingness, temporal shifts) without requiring live API network dependencies.

---

## ADR-002: Temporal Splitting Strategy & Leakage Prevention
- **Status:** Accepted
- **Context:** Time-series observations exhibit strong autocorrelation; random k-fold or train-test splits cause severe lookahead bias and data leakage.
- **Decision:** Enforce a strict chronological partition:
  - **Train:** $\le$ `2018-12-31`
  - **Validation:** `2019-01-01` to `2019-12-31`
  - **Test:** `2020-01-01` to `2020-07-01`
- **Implication:** Imputers, winsorization thresholds, and scalers are fitted exclusively on the training set. Tree models receive unscaled features, while linear models receive features scaled by the train-fitted scaler.

---

## ADR-003: Treatment of Missing Values and Unreliable Columns
- **Status:** Accepted
- **Context:** Sensor dropouts leave intermittent gaps in pollutant readings, and certain columns (e.g., Xylene) suffer >60% missingness.
- **Decision:**
  1. Drop rows missing the target (`AQI`).
  2. Drop columns exceeding 70% overall missingness (`drop_cols_over_missing`).
  3. Within each city, forward-fill gaps up to 3 days (`ffill_limit_days`).
  4. Impute any remaining nulls using the city's training-set median.
  5. Generate boolean missingness indicator columns (`<col>_was_missing`) for key pollutants.

---

## ADR-004: Baseline Model Strategy for Linked Tasks
- **Status:** Accepted
- **Context:** The system predicts next-day AQI (numeric) and next-day AQI Category (categorical).
- **Decision:**
  - Establish floor performance using simple naive baselines (Persistence, Seasonal Naive, City-Month Mean, Majority Class).
  - Train ML regressors (Ridge, Random Forest, XGBoost, LightGBM) for Task A.
  - Train ML classifiers (Logistic Regression, Random Forest, Gradient Boosted Trees) for Task B, and contrast them with the "regress-then-bucket" strategy to maintain prediction consistency.

---

## ADR-005: DVC Remote Storage Setup
- **Status:** Accepted
- **Context:** Need reproducible, local-first DVC remote storage that works offline without mandatory cloud credentials while allowing graduation to remote storage (e.g., DagsHub/S3).
- **Decision:** Configure a local file-based DVC remote (`../dvc-storage` or local directory outside git) as default, supporting cloud endpoints via `dvc remote modify --local`.

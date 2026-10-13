# Milestone 1: Dataset Approval Request Form

**Course:** AI252TA — Machine Learning Operations  
**Academic Term:** 2026–2027  
**Project Title:** AI-Based Air Quality Awareness Assistant  
**Team Composition:** 2-Person Student Group (Approved)  
- Student A: [Name / USN]
- Student B: [Name / USN]
**Date of Submission:** 2026-10-14  

---

## 1. Problem Statement & Project Scope

Urban air pollution poses severe public health challenges in Indian metropolitan centers. This project aims to build an end-to-end MLOps pipeline for predicting next-day air quality metrics:
- **Task A (Regression):** Predict numeric next-day Air Quality Index (`AQI`, range: 0–500+).
- **Task B (Classification):** Categorize next-day air quality into the 6 Indian CPCB health tiers (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`).

Milestone 1 focuses on foundational data engineering, reproducible pipelines (DVC), version control (Git), baseline modeling, and experiment tracking (MLflow).

---

## 2. Dataset Specifications

| Attribute | Specification |
|---|---|
| **Dataset Title** | Air Quality Data in India (`city_day.csv`) |
| **Originating Authority** | Central Pollution Control Board (CPCB), MoEFCC, India |
| **Source URL** | https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india |
| **License / Terms of Use** | CC0: Public Domain / Open Government Data (OGD) |
| **Dataset Dimensions** | 29,531 rows × 16 columns (2.57 MB / 2,574,056 bytes) |
| **Temporal Coverage** | January 1, 2015 – July 1, 2020 (daily frequency) |
| **Geographic Coverage** | 26 major Indian cities (including Bengaluru, Delhi, Mumbai, Chennai, etc.) |
| **Target Variable(s)** | `AQI` (Continuous numeric) & `AQI_Bucket` (CPCB 6-class categorical) |
| **Key Features** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene |

---

## 3. Justification for Selection

1. **Relevance to Indian National Standards:** The dataset directly contains CPCB-standard computed AQI values and buckets, ensuring our model aligns with official health guidance.
2. **Realistic MLOps Challenges:** The dataset contains genuine real-world characteristics: missing values, sensor dropouts, city additions over time, and time-series autocorrelation.
3. **Local Focus:** Includes continuous long-term monitoring for Bengaluru, serving as our primary regional evaluation spotlight.
4. **Feasibility:** Lightweight (~2.57 MB) and structured, enabling fast, iterative local training, DVC remote pipeline execution, and MLflow logging on standard laptop hardware.

---

## 4. Risks and Mitigation Strategies

| Identified Risk | Impact | Mitigation Strategy |
|---|---|---|
| **Temporal Data Leakage** | Optimistic / invalid evaluation metrics | Strict chronological splitting (Train $\le$ 2018, Val 2019, Test 2020). Imputers and scalers fit strictly on train split. |
| **High Missingness (e.g. Xylene, PM10)** | Degraded model performance | Forward-fill within cities (up to 3-day limit), drop high-missingness columns (>70%), impute residual values with train-period city median, and generate missingness indicators. |
| **COVID-19 Lockdown Distortions (2020)** | Out-of-distribution test set | Explicitly treat 2020 as a real-world test case; evaluate test set both with and without the Mar–Jun 2020 lockdown period. |
| **Class Imbalance (Severe vs Moderate)** | Low minority class recall | Use macro-F1, balanced class weights in tree/logistic classifiers, and evaluate "within-one-category" accuracy. |

---

## 5. Planned Split & Validation Strategy

- **Training Split:** 2015-01-01 to 2018-12-31 (~70% of chronological data, across all qualifying cities)
- **Validation Split:** 2019-01-01 to 2019-12-31 (~15% of data, used for baseline model selection and hyperparameter tuning)
- **Test Split:** 2020-01-01 to 2020-07-01 (~15% of data, touched only once for final reporting)
- **Cross-Validation:** 4-fold `TimeSeriesSplit` applied strictly within the training window.

---

## 6. Faculty Review and Sign-off

| Field | Faculty Reviewer Entry |
|---|---|
| **Reviewing Faculty Name:** | ________________________________________________ |
| **Department / Designation:** | ________________________________________________ |
| **Date of Review:** | ________________________________________________ |
| **Decision:** | [ ] APPROVED &nbsp;&nbsp;&nbsp;&nbsp; [ ] REVISE AND RESUBMIT &nbsp;&nbsp;&nbsp;&nbsp; [ ] REJECTED |
| **Faculty Comments / Directives:** | ________________________________________________ |
| **Signature:** | ________________________________________________ |

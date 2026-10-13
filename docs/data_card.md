# Dataset Card: Central Pollution Control Board (CPCB) India Air Quality (2015–2020)

## 1. Dataset Overview & Provenance

- **Dataset Name:** Air Quality Data in India (`city_day.csv`)
- **Primary Source:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India
- **Distribution Portal:** Kaggle ("Air Quality Data in India" by rohanrao) / National Data and Analytics Platform (NDAP)
- **Source URL:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- **Download Date:** 2026-10-13
- **File Name:** `city_day.csv`
- **File Size:** 2,574,056 bytes (~2.57 MB)
- **SHA-256 Checksum:** `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847`
- **License / Terms of Use:** CC0: Public Domain / Open Government Data (OGD) Platform India terms.

### Collection Method & Measurement Provenance
- **Monitoring Infrastructure:** Automated Continuous Ambient Air Quality Monitoring Stations (CAAQMS) operated by CPCB and respective State Pollution Control Boards (SPCBs).
- **Temporal Resolution:** Originally collected as hourly readings across localized stations, aggregated into 24-hour city-level daily averages.
- **Geographic Coverage:** 26 major Indian cities (including Bengaluru, Delhi, Mumbai, Kolkata, Chennai, Hyderabad, Ahmedabad, Patna, etc.).
- **Temporal Span:** January 1, 2015 to July 1, 2020 (2,009 calendar days).
- **Known vs. Unknown:**
  - *Known:* Station sensors use standard optical / chemiluminescent / beta-attenuation techniques specified by CPCB protocols.
  - *Unknown:* Specific sensor-level calibration schedules, historical sensor replacement dates, and the exact station weighting used when computing citywide averages.

---

## 2. Dataset Dimensions & Summary

- **Total Records (Rows):** 29,531 rows
- **Total Features (Columns):** 16 columns
- **Distinct Cities:** 26 cities
- **Date Range:** 2015-01-01 to 2020-07-01
- **Observation Frequency:** Daily per city
- **Duplicate (City, Date) Records:** 0
- **Target Variables:**
  - `AQI` (Numeric, Continuous, range: 13.0 to 2049.0; 4,681 missing / 15.85%)
  - `AQI_Bucket` (Categorical, 6 classes per CPCB standard; 4,681 missing / 15.85%)

---

## 3. Data Dictionary

| Column Name | Data Type | Physical Unit | Description | Missing Count | % Missing |
|---|---|---|---|---|---|
| `City` | String / Object | N/A | Name of the Indian city | 0 | 0.00% |
| `Date` | Date (`YYYY-MM-DD`) | N/A | Observation date (calendar day) | 0 | 0.00% |
| `PM2.5` | Float64 | $\mu g/m^3$ | Fine particulate matter ($\le 2.5 \mu m$) | 4,598 | 15.57% |
| `PM10` | Float64 | $\mu g/m^3$ | Respirable particulate matter ($\le 10 \mu m$) | 11,140 | 37.72% |
| `NO` | Float64 | $\mu g/m^3$ | Nitric Oxide | 3,582 | 12.13% |
| `NO2` | Float64 | $\mu g/m^3$ | Nitrogen Dioxide | 3,585 | 12.14% |
| `NOx` | Float64 | $ppb$ / $\mu g/m^3$ | Nitrogen Oxides (total) | 4,185 | 14.17% |
| `NH3` | Float64 | $\mu g/m^3$ | Ammonia | 10,328 | 34.97% |
| `CO` | Float64 | $mg/m^3$ | Carbon Monoxide | 2,059 | 6.97% |
| `SO2` | Float64 | $\mu g/m^3$ | Sulphur Dioxide | 3,854 | 13.05% |
| `O3` | Float64 | $\mu g/m^3$ | Ground-level Ozone | 4,022 | 13.62% |
| `Benzene` | Float64 | $\mu g/m^3$ | Benzene volatile organic compound | 5,623 | 19.04% |
| `Toluene` | Float64 | $\mu g/m^3$ | Toluene volatile organic compound | 8,041 | 27.23% |
| `Xylene` | Float64 | $\mu g/m^3$ | Xylene volatile organic compound | 18,109 | 61.32% |
| `AQI` | Float64 | Index (0–500+) | Official Indian Air Quality Index | 4,681 | 15.85% |
| `AQI_Bucket` | String / Object | N/A | CPCB standard classification bucket | 4,681 | 15.85% |

---

## 4. CPCB AQI Classification Scale

| AQI Band | Category | Associated Health Guidance |
|---|---|---|
| 0–50 | **Good** | Minimal impact |
| 51–100 | **Satisfactory** | Minor breathing discomfort to sensitive people |
| 101–200 | **Moderate** | Breathing discomfort to people with lungs, asthma, and heart diseases |
| 201–300 | **Poor** | Breathing discomfort to most people on prolonged exposure |
| 301–400 | **Very Poor** | Respiratory illness on prolonged exposure |
| 401–500 | **Severe** | Affects healthy people and seriously impacts those with existing diseases |

---

## 5. Known Limitations & Data Quality Issues

1. **Missingness:** High missingness rates across volatile organics (e.g., Xylene ~61%) and PM10 (~38%). Cities that entered the CAAQMS network later in the timeline have large initial gaps.
2. **Missing Target (`AQI`):** Approximately 16% of rows have null AQI because CPCB guidelines require at least 3 pollutants (with at least one being PM2.5 or PM10) to compute a valid sub-index. Rows with missing target cannot be used for supervised training/evaluation.
3. **Temporal Anomaly (COVID-19 Lockdown):** March 2020 through June 2020 experienced nationwide lockdowns, creating sudden structural shifts in urban traffic, industrial emissions, and air quality patterns. This period represents an exogenous shock rather than normal trend.
4. **Lack of Meteorological Covariates:** The dataset lacks boundary layer height, temperature, precipitation, wind speed, and humidity, which heavily influence pollutant dispersion and concentration.
5. **Spatial Aggregation:** A single city average hides intra-urban microclimates (e.g., industrial clusters vs. residential parks).

---

## 6. Ethical & Privacy Considerations

- **Human Subjects / PII:** The dataset contains purely environmental physical measurements. No personally identifiable information (PII) or human subject tracking is involved.
- **Fairness & Dual-Use:** Air quality reporting has direct socioeconomic relevance. Models trained on public data must not misrepresent air pollution risks to vulnerable communities.

---

## 7. Intended vs. Non-Intended Use

- **Intended Use:** Academic and research benchmarking of machine learning models for short-term (next-day) air quality forecasting, baseline comparison, and MLOps lifecycle validation.
- **Non-Intended / Prohibited Use:** Clinical diagnostic decisions, regulatory enforcement compliance, or safety-critical emergency evacuations without human-in-the-loop validation and official CAAQMS station calibration.

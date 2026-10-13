# Dataset Card: Central Pollution Control Board (CPCB) India Air Quality (2015–2020)

## 1. Dataset Overview & Provenance

- **Dataset Name:** Air Quality Data in India (`city_day.csv`)
- **Primary Source:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India
- **Distribution Portal:** Kaggle ("Air Quality Data in India" by rohanrao) / National Data and Analytics Platform (NDAP)
- **Source URL:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- **Download Date:** [TO BE FILLED BY STUDENT UPON DOWNLOAD - e.g. 2026-10-14]
- **File Name:** `city_day.csv`
- **File Size:** ~2.1 MB
- **SHA-256 Checksum:** `[TO BE FILLED BY STUDENT - Run: certutil -hashfile data/raw/city_day.csv SHA256]`
- **License / Terms of Use:** Data open under CC0: Public Domain / Open Government Data (OGD) Platform India terms. (Verify exact terms on source page).

### Collection Method & Measurement Provenance
- **Monitoring Infrastructure:** Automated Continuous Ambient Air Quality Monitoring Stations (CAAQMS) operated by CPCB and respective State Pollution Control Boards (SPCBs).
- **Temporal Resolution:** Originally collected as hourly readings across localized stations, aggregated into 24-hour city-level daily averages.
- **Geographic Coverage:** 26 major Indian cities (including Bengaluru, Delhi, Mumbai, Kolkata, Chennai, Hyderabad, Ahmedabad, Patna, etc.).
- **Temporal Span:** January 1, 2015 to July 1, 2020 (~2,009 days).
- **Known vs. Unknown:**
  - *Known:* Station sensors use standard optical / chemiluminescent / beta-attenuation techniques specified by CPCB protocols.
  - *Unknown:* Specific sensor-level calibration schedules, historical sensor replacement dates, and the exact station weighting used when computing citywide averages.

---

## 2. Dataset Dimensions & Summary

- **Total Records (Rows):** ~29,531 rows
- **Total Features (Columns):** 16 columns
- **Observation Frequency:** Daily per city
- **Target Variables:**
  - `AQI` (Numeric, Continuous, range: 0–500+)
  - `AQI_Bucket` (Categorical, 6 classes per CPCB standard)

---

## 3. Data Dictionary

| Column Name | Data Type | Physical Unit | Description | Expected % Missing (Approx) |
|---|---|---|---|---|
| `City` | Categorical / String | N/A | Name of the Indian city | 0.0% |
| `Date` | Date (`YYYY-MM-DD`) | N/A | Observation date (calendar day) | 0.0% |
| `PM2.5` | Float | $\mu g/m^3$ | Fine particulate matter ($\le 2.5 \mu m$) | ~15% |
| `PM10` | Float | $\mu g/m^3$ | Respirable particulate matter ($\le 10 \mu m$) | ~38% |
| `NO` | Float | $\mu g/m^3$ | Nitric Oxide | ~12% |
| `NO2` | Float | $\mu g/m^3$ | Nitrogen Dioxide | ~12% |
| `NOx` | Float | $ppb$ / $\mu g/m^3$ | Nitrogen Oxides (total) | ~14% |
| `NH3` | Float | $\mu g/m^3$ | Ammonia | ~35% |
| `CO` | Float | $mg/m^3$ | Carbon Monoxide | ~7% |
| `SO2` | Float | $\mu g/m^3$ | Sulphur Dioxide | ~13% |
| `O3` | Float | $\mu g/m^3$ | Ground-level Ozone | ~14% |
| `Benzene` | Float | $\mu g/m^3$ | Benzene volatile organic compound | ~19% |
| `Toluene` | Float | $\mu g/m^3$ | Toluene volatile organic compound | ~27% |
| `Xylene` | Float | $\mu g/m^3$ | Xylene volatile organic compound | ~61% |
| `AQI` | Float | Index (0–500+) | Official Indian Air Quality Index | ~16% |
| `AQI_Bucket` | Categorical / String | N/A | CPCB standard classification bucket | ~16% |

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

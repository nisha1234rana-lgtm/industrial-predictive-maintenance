# Industrial Equipment Predictive Maintenance and Failure Analysis

An end-to-end predictive maintenance project using **7.1M+ real industrial sensor readings** from the MetroPT2 air-production system.

The project turns raw IoT telemetry into anomaly alerts, failure-risk predictions, maintenance priorities, cost estimates and model-monitoring outputs.

## Project Highlights

- 7,116,940 raw sensor readings
- 92 days of industrial telemetry
- DuckDB data pipeline
- 1-minute time-series aggregation
- 6h, 12h and 24h failure targets
- Isolation Forest anomaly detection
- Logistic Regression, Random Forest, XGBoost and LightGBM
- SHAP model explanations
- maintenance priority scoring
- downtime and cost simulation
- PSI and KS drift monitoring
- Streamlit maintenance dashboard

## Key Results

| Metric | Result |
|---|---:|
| Raw readings | 7,116,940 |
| Minute-level rows | 117,682 |
| Documented failures | 2 |
| Normal anomaly alert rate | 1.42% |
| Final 6h anomaly alert rate | 40.69% |
| Earliest High-risk warning | 22.84 hours before failure |
| High/Critical alert episodes | 14 |
| Failure-related episodes | 2 |
| Modeled downtime avoided | 8 hours |
| Modeled cost avoided | $20,000 |

## Maintenance Dashboard

### Equipment Health

![Equipment Health](images/equipment%20health.png)

Tracks maintenance priority, failure risk, anomaly severity and major sensor behavior.

### Failure Intelligence

![Failure Intelligence](images/failure%20intelligence.png)

Shows how early-warning and near-failure risk changed before the July equipment failure.

### Model Performance

![Model Performance](images/model%20performance.png)

Compares Logistic Regression, Random Forest, XGBoost and LightGBM across multiple failure horizons.

### Monitoring

![Monitoring](images/monitoring.png)

Tracks feature drift, anomaly behavior and weekly production-model activity.

### Business Impact

![Business Impact](images/business%20impact.png)

Compares reactive and predictive maintenance under explicit cost assumptions.

## Dataset

The project uses the **MetroPT2 industrial predictive-maintenance dataset**.

Dataset source:

https://zenodo.org/records/7766691

The raw data contains approximately one-second telemetry from April to July 2022.

Important signals include:

- compressor pressure
- pneumatic-panel pressure
- reservoir pressure
- oil temperature
- airflow
- motor current
- compressor activity
- pressure-switch state
- oil-level state

Two documented failures are included:

```text
Air leak
June 4, 2022

Oil leak
July 11–14, 2022
```

## Pipeline

```text
Raw IoT Telemetry
        ↓
DuckDB Ingestion
        ↓
1-Minute Sensor Aggregation
        ↓
Rolling Time-Series Features
        ↓
Failure Window Creation
        ↓
Anomaly Detection
        ↓
Failure Prediction
        ↓
SHAP Explainability
        ↓
Maintenance Priority
        ↓
Cost Simulation
        ↓
Drift Monitoring
        ↓
Streamlit Dashboard
```

## Feature Engineering

Second-level telemetry was reduced to approximately **117K minute-level observations**.

Features include:

- 15-minute, 1-hour and 6-hour rolling averages
- rolling volatility
- temperature trends
- motor-current instability
- pressure deviation
- airflow deviation
- rolling z-scores
- compressor activity
- data-coverage metrics

Failure targets were created for:

```text
Failure within 6 hours
Failure within 12 hours
Failure within 24 hours
```

Actual failure periods and post-failure recovery periods were excluded from normal model training.

## Anomaly Detection

Isolation Forest was trained only on an early healthy operating period.

```text
Healthy baseline alert rate       1.00%
Normal-period alert rate          1.42%
24h pre-failure alert rate       10.60%
Final 6h alert rate              40.69%
```

The increase in anomalies closer to failure shows that multivariate sensor behavior became increasingly abnormal.

## Failure Prediction

Models compared:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

A chronological split was used instead of random train/test sampling.

```text
Training
April 28 – June 14

Calibration
June 15 – June 30

Final Test
July 1 – July 28
```

The July oil-leak failure was therefore unseen during model training.

### 24-Hour Early Warning

The Logistic Regression model provided the most useful long-range warning.

```text
PR-AUC                 0.2021
ROC-AUC                0.7139
Recall                 27.81%
False-alert rate        6.04%
Earliest warning       22.84 hours
```

The first High-risk alert occurred at:

```text
2022-07-10 11:20
```

The documented failure began the following day.

### Near-Failure Detection

The 6-hour Random Forest produced:

```text
PR-AUC                 1.0000
Recall                 100%
False-alert rate        1.42%
```

However, only **43 usable observations** existed immediately before the second failure.

Its earliest warning was only:

```text
0.71 hours before failure
```

For that reason, this model is treated as a **near-failure detector**, not proof of six-hour advance prediction.

## SHAP Explainability

Important 24-hour warning features included:

- TP3 pressure 6-hour baseline
- reservoir pressure 6-hour baseline
- motor-current 6-hour average
- motor-current instability
- anomaly score

Important near-failure features included:

- motor-current behavior
- compressor activity
- airflow deviation
- pressure-switch state
- oil-level state

At the first near-failure warning, airflow was about **21.2 units above its six-hour baseline**.

## Maintenance Decision Engine

The maintenance score combines:

```text
45%  24-hour early-warning signal
35%  near-failure signal
20%  anomaly severity
```

Risk levels:

```text
Low
Medium
High
Critical
```

Example output:

```text
Risk Level
High

Primary Warning Signal
Motor current instability

Recommended Action
Inspect during the next available maintenance window
```

The final test period contained:

```text
14 High/Critical alert episodes
2 failure-related episodes
12 false alert episodes
```

## Business Impact Simulation

MetroPT2 does not contain real maintenance costs, so the business values below are scenario assumptions.

### Reactive Maintenance

```text
12 hours downtime
$2,500 per downtime hour
$12,000 emergency repair

Total
$42,000
```

### Predictive Maintenance

```text
4 hours controlled downtime
$5,000 planned repair
$500 inspection cost per alert episode

Total
$22,000
```

Modeled impact:

```text
Downtime avoided       8 hours
Cost avoided           $20,000
```

## Model Monitoring

Production monitoring uses:

- Population Stability Index
- Kolmogorov-Smirnov statistics
- anomaly distributions
- weekly alert rates
- weekly prediction probabilities

Results:

```text
Stable features        5
Moderate drift         3
High drift             7
```

Monitoring status:

```text
Retraining review recommended
```

High drift is treated as a review signal rather than automatic model failure because the production period contains changing equipment behavior and an actual failure.

## Technology

```text
Python
SQL
DuckDB
Pandas
NumPy
Scikit-learn
XGBoost
LightGBM
SHAP
SciPy
Plotly
Streamlit
PyArrow
```

## Project Structure

```text
Industrial Equipment Predictive Maintenance
│
├── app
│   └── maintenance console.py
│
├── images
├── models
├── reports
├── src
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Run the Project

Download `MetroPT2.csv` from:

https://zenodo.org/records/7766691

Place it inside:

```text
data/raw/metropt2/MetroPT2.csv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline scripts inside `src`, then launch the dashboard:

```bash
streamlit run "app/maintenance console.py"
```

## Important Limitation

MetroPT2 contains only two documented failure events, so model results should be interpreted as a case study rather than proof of general performance across many machines or failure types.

The project deliberately keeps this limitation visible instead of overstating the model results.
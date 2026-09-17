# Industrial Equipment Predictive Maintenance and Failure Analysis

An end-to-end industrial analytics and machine learning project that uses real equipment telemetry to detect abnormal operating behavior, predict upcoming failures, prioritize maintenance actions and estimate the operational impact of predictive maintenance.

The project processes more than **7.1 million real IoT sensor readings** from an industrial air-production system and transforms raw second-level telemetry into a complete predictive-maintenance workflow.

The final solution combines:

- IoT sensor data ingestion
- DuckDB analytics warehouse
- time-series feature engineering
- anomaly detection
- 6-hour, 12-hour and 24-hour failure-risk modeling
- Logistic Regression, Random Forest, XGBoost and LightGBM
- SHAP explainability
- maintenance priority scoring
- recommended maintenance actions
- downtime and maintenance-cost simulation
- model and sensor drift monitoring
- interactive Streamlit maintenance console

---

## Project Overview

Industrial equipment generates large amounts of telemetry, but raw sensor readings alone do not tell maintenance teams when equipment is beginning to deteriorate.

This project builds a system that converts equipment telemetry into operational decisions.

The workflow answers questions such as:

- Is the equipment operating abnormally?
- Are pressure, temperature, airflow or motor-current patterns changing?
- Is a failure likely within the next 6, 12 or 24 hours?
- Which signals are contributing most to the warning?
- How urgent is the maintenance action?
- What maintenance action should be taken?
- How much downtime could predictive maintenance potentially avoid?
- Has sensor behavior shifted enough that the model should be reviewed?

The result is a complete predictive-maintenance system rather than only a machine learning model.

---

# Key Results

| Metric | Result |
|---|---:|
| Raw sensor readings | 7,116,940 |
| Observation period | 92 calendar days |
| Original sensor fields | 21 |
| Minute-level analytical rows | 117,682 |
| Documented failures | 2 |
| Healthy anomaly baseline alert rate | 1.00% |
| Normal-period anomaly rate | 1.42% |
| 24-hour pre-failure anomaly rate | 10.60% |
| Final 6-hour anomaly rate | 40.69% |
| Earliest High/Critical warning | 22.84 hours before failure |
| High/Critical maintenance episodes | 14 |
| Failure-related alert episodes | 2 |
| False alert episodes | 12 |
| Modeled downtime avoided | 8 hours |
| Modeled cost avoided | $20,000 |
| High-drift monitored features | 7 |

---

# Maintenance Console

The final Streamlit application brings the entire analytics workflow together into one maintenance decision console.

## Equipment Health

![Equipment Health](images/equipment%20health.png)

The Equipment Health page provides the current operational view of the system.

It includes:

- equipment risk level
- maintenance priority score
- 24-hour failure score
- anomaly percentile
- primary warning signal
- recommended maintenance action
- maintenance priority timeline
- oil-temperature behavior
- motor-current behavior
- pressure behavior
- airflow deviation

Example output:

```text
Risk Level
High

Maintenance Priority
72.4 / 100

24h Failure Risk
91.7%

Primary Warning Signal
Motor current instability

Recommended Action
Inspect during the next available maintenance window
```

---

## Failure Intelligence

![Failure Intelligence](images/failure%20intelligence.png)

This page focuses on how risk develops before a known equipment failure.

The system uses two complementary models.

### Early-warning model

A 24-hour Logistic Regression model is used for longer-range warning.

It generated a High-risk alert approximately:

```text
22.84 hours before the July failure
```

### Near-failure detector

A Random Forest model identifies highly abnormal equipment behavior immediately before failure.

The earliest usable warning from this model occurred approximately:

```text
0.71 hours before failure
```

The page also displays:

- early-warning probability
- near-failure probability
- model thresholds
- alert episodes
- failure-related alerts
- false alert episodes
- major failure drivers

---

## Model Performance

![Model Performance](images/model%20performance.png)

The Model Performance page compares four supervised models across multiple failure horizons.

Models evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Prediction horizons:

- 6 hours
- 12 hours
- 24 hours

Evaluation focuses on metrics appropriate for rare-event prediction:

- PR-AUC
- ROC-AUC
- precision
- recall
- F1
- false-alert rate
- failure-window detection rate
- earliest warning time

Accuracy is intentionally not treated as the primary metric because equipment failures are rare.

---

## Model Monitoring

![Monitoring](images/monitoring.png)

The monitoring layer evaluates whether operating conditions in later periods differ from the healthy training baseline.

It tracks:

- Population Stability Index
- Kolmogorov-Smirnov statistics
- sensor distribution changes
- anomaly-score behavior
- weekly prediction probabilities
- weekly alert rates
- failure-risk minutes
- retraining-review triggers

The monitoring system identified:

```text
Stable features          5
Moderate drift           3
High drift               7
```

Current monitoring recommendation:

```text
Retraining review recommended
```

This does not automatically mean the model failed.

The later production period also contains a documented failure and changing equipment behavior, so some drift is expected and operationally meaningful.

---

## Business Impact

![Business Impact](images/business%20impact.png)

The final layer translates predictive-maintenance signals into a simple operational cost simulation.

The simulation compares:

```text
Reactive Maintenance
vs
Predictive Maintenance
```

Under the defined scenario assumptions, predictive maintenance reduces modeled downtime by:

```text
8 hours
```

and modeled total maintenance impact by:

```text
$20,000
```

The cost assumptions are explicitly modeled scenarios and are not actual costs supplied by the MetroPT2 dataset.

---

# Dataset

The project uses the **MetroPT2 industrial predictive-maintenance dataset**.

MetroPT2 contains telemetry from an industrial air-production system and was designed for research involving:

- anomaly detection
- predictive maintenance
- failure prediction
- equipment condition monitoring

Dataset source:

https://zenodo.org/records/7766691

---

## Dataset Scale

```text
Sensor readings      7,116,940
Sensor fields        21
Start                April 28, 2022
End                  July 28, 2022
Calendar days        92
Sampling             approximately 1 second
Missing values       0
Duplicate timestamps 0
```

---

# Sensor Data

The dataset contains continuous and digital equipment signals.

Important sensors used in the project include:

| Sensor | Description |
|---|---|
| TP2 | Compressor pressure |
| TP3 | Pneumatic-panel pressure |
| H1 | Pressure-related operating signal |
| DV pressure | Pressure measurement |
| Reservoirs | Air-tank pressure |
| Oil Temperature | Compressor oil temperature |
| Flowmeter | Airflow |
| Motor Current | Compressor motor electrical current |
| COMP | Compressor operating signal |
| DV Electric | Electrical valve state |
| Towers | Air-dryer tower state |
| MPG | Operating signal |
| LPS | Low-pressure warning |
| Pressure Switch | Pressure-switch status |
| Oil Level | Oil-level status |
| Caudal Impulses | Flow-related digital signal |
| GPS Speed | Equipment movement information |

GPS fields were retained during raw ingestion but were not central to the predictive-maintenance models.

---

# Documented Failure Events

Two documented equipment failure periods were incorporated into the analytical warehouse.

## Failure 1

```text
Failure Type
Air Leak

Start
June 4, 2022 10:19

End
June 4, 2022 14:22

Approximate Duration
4.05 hours
```

## Failure 2

```text
Failure Type
Oil Leak

Start
July 11, 2022 10:10

End
July 14, 2022 10:22

Approximate Duration
72.20 hours
```

These failure events were used to construct supervised prediction targets.

---

# Project Architecture

```text
Raw MetroPT2 Sensor Data
        |
        v
DuckDB Raw Warehouse
        |
        v
Sensor Quality Validation
        |
        v
1-Minute Time Aggregation
        |
        v
6h / 12h / 24h Failure Windows
        |
        v
Rolling Time-Series Features
        |
        +----------------------------+
        |                            |
        v                            v
Isolation Forest              Supervised ML
Anomaly Detection             Failure Prediction
        |                            |
        +-------------+--------------+
                      |
                      v
               SHAP Explainability
                      |
                      v
             Maintenance Priority
                      |
                      v
            Recommended Action
                      |
                      v
        Downtime and Cost Simulation
                      |
                      v
           Drift / Model Monitoring
                      |
                      v
          Streamlit Maintenance Console
```

---

# Data Engineering

## Raw Ingestion

The original CSV is approximately 1.2 GB and contains more than seven million records.

Instead of loading the entire dataset into memory with pandas, DuckDB is used for efficient ingestion and analytical processing.

The raw warehouse contains:

```text
7,116,940 rows
0 missing sensor values
0 duplicate timestamps
```

The database preserves the original sensor observations while standardizing field names for downstream analysis.

---

# Minute-Level Analytical Layer

Raw telemetry is recorded approximately once per second.

For modeling and operational analytics, the data is aggregated into one-minute windows.

This reduces millions of second-level observations into:

```text
117,682 minute-level rows
```

Average sensor readings per minute:

```text
60.48
```

For each minute, statistical features include:

- mean
- minimum
- maximum
- standard deviation
- equipment-state activity rate

This produces a more practical modeling grain while retaining equipment behavior.

---

# Failure Window Engineering

Three supervised targets were created:

```text
failure_next_6h
failure_next_12h
failure_next_24h
```

For each minute, the pipeline determines whether a documented failure begins within the corresponding future time window.

Additional labels include:

```text
failure_active
post_failure_24h
hours_to_next_failure
```

Actual failure periods are excluded from normal model training.

The first 24 hours after each failure are also excluded to avoid treating post-failure or recovery behavior as healthy operation.

---

# Failure Label Distribution

```text
Active failure minutes        3,864
Failure next 6h                 691
Failure next 12h              1,230
Failure next 24h              2,425
Post-failure excluded         1,976
Eligible modeling minutes   111,842
```

---

# Time-Series Feature Engineering

Predictive maintenance depends more on changing behavior than on individual sensor values.

The project therefore creates rolling historical features using only information available before each prediction timestamp.

No future sensor information is used.

---

## Rolling Windows

Features are calculated across:

```text
15 minutes
1 hour
6 hours
```

Examples include:

- rolling mean
- rolling standard deviation
- rolling maximum
- short-term trend
- long-term trend
- deviation from 6-hour baseline
- rolling z-score
- sensor instability
- operational-state change

---

# Temperature Features

Examples include:

```text
oil_temp_avg_15m
oil_temp_avg_1h
oil_temp_avg_6h

oil_temp_std_1h
oil_temp_std_6h

oil_temp_short_trend
oil_temp_long_trend

oil_temp_deviation_6h
oil_temp_zscore_6h
```

Before failure, oil-temperature behavior showed stronger trending and instability.

Normal vs pre-failure comparison:

```text
Oil temperature trend

Normal                  0.254
Within 24h of failure   0.762
```

Oil-temperature instability:

```text
Normal                  2.088
Within 24h of failure   2.588
```

---

# Motor Current Features

Motor-current behavior is an important indicator of changing equipment load.

Features include:

```text
motor_current_avg_15m
motor_current_avg_1h
motor_current_avg_6h

motor_current_std_1h
motor_current_std_6h

motor_current_short_trend
motor_current_long_trend

motor_current_zscore_6h
```

Motor-current variables became some of the most important predictors in both supervised models.

---

# Pressure Features

Pressure analytics include:

```text
tp3_avg_15m
tp3_avg_1h
tp3_avg_6h

tp3_std_1h
tp3_std_6h

tp3_short_trend
tp3_long_trend
tp3_zscore_6h
```

Reservoir pressure features include:

```text
reservoir_avg_1h
reservoir_avg_6h

reservoir_std_1h
reservoir_std_6h

reservoir_deviation_6h
reservoir_zscore_6h
```

A pressure-balance feature was also created:

```text
ABS(TP3 pressure - reservoir pressure)
```

This helps capture abnormal relationships between parts of the pneumatic system.

---

# Airflow Features

Airflow became especially important immediately before the July failure.

Features include:

```text
flow_avg_1h
flow_avg_6h
flow_std_1h
flow_std_6h
flow_deviation_6h
flow_zscore_6h
```

Average airflow instability:

```text
Normal                  0.000
Within 24h of failure   0.543
```

At the first near-failure warning, airflow was approximately:

```text
21.2 units above its six-hour baseline
```

---

# Compressor Activity Features

Digital compressor signals were aggregated into operating-state rates.

Examples include:

```text
compressor_activity_1h
compressor_activity_6h
compressor_activity_change
```

These features help distinguish normal equipment cycles from abnormal equipment behavior.

---

# Data Coverage Features

Real sensor data contains operational gaps.

The project tracks:

```text
coverage_1h
coverage_6h
```

Average data coverage:

```text
1-hour coverage    98.6%
6-hour coverage    94.3%
```

Rows with at least 80% coverage across the six-hour historical window:

```text
106,346
```

Low-quality windows are excluded from modeling.

---

# Anomaly Detection

An unsupervised Isolation Forest model was trained to identify unusual combinations of sensor behavior.

The anomaly model uses 15 engineered features including:

- oil-temperature trends
- motor-current trends
- pressure instability
- reservoir-pressure behavior
- airflow behavior
- compressor activity
- pressure balance

---

## Healthy Baseline

The Isolation Forest was trained only on an early healthy period:

```text
April 28, 2022
to
June 2, 2022
```

Training observations:

```text
43,055
```

This prevents known failure behavior from being learned as normal equipment operation.

---

# Anomaly Threshold

The alert threshold is based on the 99th percentile of healthy baseline anomaly scores.

Healthy baseline alert rate:

```text
1.00%
```

This creates a data-driven alert threshold instead of using an arbitrary anomaly score.

---

# Anomaly Detection Results

Normal operation:

```text
Anomaly alert rate
1.42%
```

Within 24 hours of failure:

```text
Anomaly alert rate
10.60%
```

Within 12 hours:

```text
17.41%
```

Within the final usable 6-hour target window:

```text
40.69%
```

This suggests abnormal equipment behavior becomes substantially more frequent as failure approaches.

---

# Supervised Failure Prediction

Four supervised algorithms were evaluated:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

Models were built separately for:

```text
6-hour failure horizon
12-hour failure horizon
24-hour failure horizon
```

---

# Chronological Validation

A random train/test split was intentionally avoided.

Using random rows would allow observations from the same operating periods to appear in both training and test sets and could dramatically overstate model performance.

Instead, the project uses chronological evaluation.

```text
Training

April 28, 2022
to
June 14, 2022

56,151 rows
```

```text
Calibration

June 15, 2022
to
June 30, 2022

18,655 rows
```

```text
Final Test

July 1, 2022
to
July 28, 2022

26,282 rows
```

The July oil-leak failure therefore acts as an unseen future failure during final testing.

---

# Model Evaluation Metrics

Because failure observations are rare, plain accuracy is not useful.

The project evaluates:

- Precision-Recall AUC
- ROC-AUC
- precision
- recall
- F1
- failure-window alert coverage
- false-alert rate
- earliest warning time

The decision threshold is calibrated using normal observations from the calibration period.

The threshold approximately represents a 1% false-alert budget under calibration conditions.

---

# Model Comparison

## 6-Hour Horizon

| Model | PR-AUC | ROC-AUC | Recall | False Alert Rate |
|---|---:|---:|---:|---:|
| Random Forest | 1.0000 | 1.0000 | 100.00% | 1.42% |
| Logistic Regression | 0.5788 | 0.9928 | 86.05% | 1.55% |
| XGBoost | 0.0547 | 0.9865 | 72.09% | 1.43% |
| LightGBM | 0.0415 | 0.9815 | 0.00% | 1.17% |

The Random Forest appears extremely strong based on row-level metrics.

However, the final test failure had only:

```text
43 usable observations
```

inside the nominal six-hour window.

The earliest usable observation was only:

```text
0.71 hours before failure
```

Therefore this model is described as a **near-failure detector**, not a six-hour early-warning model.

---

# 12-Hour Horizon

Performance at the 12-hour horizon was substantially weaker.

The Logistic Regression model achieved:

```text
PR-AUC      0.1991
ROC-AUC     0.4517
Recall      19.37%
```

The 12-hour models are retained for model comparison but are not used as the main operational models.

---

# 24-Hour Early Warning

The 24-hour Logistic Regression model provided the most useful long-range maintenance warning.

Results:

```text
PR-AUC                       0.2021
ROC-AUC                      0.7139
Precision                    14.61%
Recall                       27.81%
F1                           0.1916
False-alert rate              6.04%
Failure-window alert rate    27.81%
```

The earliest High-risk warning occurred:

```text
22.84 hours before failure
```

The first warning timestamp was:

```text
July 10, 2022 11:20
```

The documented failure began:

```text
July 11, 2022 10:10
```

---

# Two-Model Maintenance Strategy

The final decision layer therefore does not rely on a single model.

Instead it uses two complementary models.

## Early-Warning Model

```text
Model
Logistic Regression

Horizon
24 hours

Purpose
Provide maintenance teams with advance notice

Earliest observed warning
22.84 hours
```

## Near-Failure Detector

```text
Model
Random Forest

Nominal target
6 hours

Practical interpretation
Near-failure detection

Earliest usable warning
0.71 hours
```

This allows the system to distinguish between:

```text
Longer-range deterioration
and
Immediate failure behavior
```

---

# Model Explainability

SHAP is used to understand the signals influencing model predictions.

The purpose is not to claim that individual sensors directly cause failure.

SHAP explains how the trained model responds to different sensor patterns.

This distinction is important because many rolling pressure and motor-current variables are correlated.

---

# 24-Hour Early-Warning Drivers

The strongest SHAP features included:

```text
TP3 pressure 6-hour average
Reservoir pressure 6-hour average
Motor current 6-hour average
Reservoir pressure 1-hour average
TP3 pressure 1-hour average
Motor current 6-hour instability
Motor current long-term trend
Anomaly score
Motor current level
Motor current z-score
```

The first 24-hour warning occurred at:

```text
2022-07-10 11:20
```

Time before failure:

```text
22.84 hours
```

Failure probability:

```text
91.73%
```

Alert threshold:

```text
49.90%
```

---

# Near-Failure Drivers

Important Random Forest features included:

```text
Motor current 6-hour average
Compressor activity 6-hour average
Motor current 6-hour instability
Pressure-switch state
Oil-level state
Airflow deviation
Current airflow
1-hour airflow average
6-hour airflow instability
1-hour airflow instability
```

The first near-failure warning occurred at:

```text
2022-07-11 09:28
```

Time before failure:

```text
0.71 hours
```

The most interesting operational signal was:

```text
Airflow deviation from 6-hour baseline
21.20
```

---

# Maintenance Priority Engine

Machine learning probabilities alone are difficult for maintenance teams to use.

A decision layer converts model outputs into a maintenance priority score.

The score combines:

```text
45%   24-hour early-warning signal
35%   near-failure signal
20%   anomaly severity
```

The final score ranges from:

```text
0 to 100
```

---

# Risk Levels

Equipment observations are classified into four operational categories:

```text
Low
Medium
High
Critical
```

### Low

Normal operation.

Recommended action:

```text
Continue normal operation and routine monitoring
```

### Medium

Sensor behavior is becoming unusual.

Recommended action:

```text
Increase monitoring and inspect if elevated risk persists
```

### High

The early-warning model indicates elevated failure risk.

Recommended action:

```text
Inspect during the next available maintenance window
```

### Critical

The near-failure detector or a combination of supervised and anomaly signals indicates urgent risk.

Recommended action:

```text
Inspect immediately and prepare controlled maintenance shutdown
```

---

# Primary Warning Signal

The decision layer identifies the dominant operational warning signal from:

- motor-current instability
- airflow deviation
- oil-temperature trend
- TP3 pressure deviation
- reservoir-pressure deviation

This creates a more understandable maintenance message than returning only a probability.

Example:

```text
Failure risk
High

Primary warning signal
Airflow deviation

Recommended action
Inspect during the next maintenance window
```

---

# Maintenance Alert Episodes

Consecutive High/Critical observations are grouped into maintenance episodes.

This prevents hundreds of consecutive minute-level warnings from being treated as hundreds of separate maintenance requests.

A new episode begins when more than 60 minutes pass between High/Critical warnings.

Final test-period results:

```text
High/Critical episodes      14
Failure-related episodes     2
False alert episodes        12
```

This gives a more realistic view of operational workload.

---

# Downtime and Maintenance Cost Simulation

MetroPT2 does not provide actual financial maintenance data.

Therefore all costs are explicitly defined as scenario assumptions.

They are used to demonstrate how model outputs can be translated into business impact.

---

# Reactive Maintenance Scenario

Assumptions:

```text
Unplanned downtime
12 hours

Downtime cost
$2,500 per hour

Emergency repair
$12,000
```

Downtime cost:

```text
12 × $2,500
=
$30,000
```

Total modeled reactive cost:

```text
$42,000
```

---

# Predictive Maintenance Scenario

Assumptions:

```text
Controlled downtime
4 hours

Downtime cost
$2,500 per hour

Planned repair
$5,000

Inspection cost
$500 per High/Critical episode
```

With 14 alert episodes:

```text
Inspection cost
14 × $500
=
$7,000
```

Total modeled predictive-maintenance cost:

```text
$22,000
```

---

# Modeled Business Impact

Under these assumptions:

```text
Reactive cost        $42,000
Predictive cost      $22,000

Modeled savings      $20,000
```

Modeled downtime reduction:

```text
Reactive downtime     12 hours
Predictive downtime    4 hours

Downtime avoided       8 hours
```

These values are scenario-modeling outputs, not actual observed financial outcomes from the dataset.

---

# Model Monitoring

Predictive-maintenance systems need monitoring after deployment because equipment behavior can change over time.

The project therefore includes a production-style monitoring layer.

---

# Population Stability Index

PSI is used to compare each monitored feature against the healthy baseline distribution.

Interpretation:

```text
PSI < 0.10
Stable

PSI 0.10 to 0.25
Moderate drift

PSI > 0.25
High drift
```

---

# Drift Results

Largest observed shifts included:

| Feature | PSI | Drift |
|---|---:|---|
| Oil temperature 1h instability | 4.8581 | High |
| Motor current 6h average | 1.1034 | High |
| Motor current 6h instability | 1.0821 | High |
| Motor current mean | 0.5962 | High |
| Anomaly percentile | 0.5301 | High |
| Anomaly score | 0.5300 | High |
| Oil temperature mean | 0.3072 | High |
| TP3 6h pressure | 0.2419 | Moderate |
| Reservoir 6h pressure | 0.2335 | Moderate |
| Oil temperature long trend | 0.1744 | Moderate |

Summary:

```text
Stable features        5
Moderate drift         3
High drift             7
```

Monitoring status:

```text
Retraining review recommended
```

---

# Drift Interpretation

High drift does not automatically mean the model is broken.

The July production period contains:

- an actual documented equipment failure
- different operating behavior
- increased anomaly activity

Part of the observed distribution shift may therefore be exactly the abnormal behavior the predictive-maintenance system is designed to identify.

For this reason the application recommends:

```text
Retraining review
```

rather than automatically retraining or declaring model failure.

---

# Weekly Monitoring

The project also tracks weekly production behavior.

Metrics include:

- number of observations
- mean failure probability
- maximum failure probability
- anomaly score
- anomaly percentile
- true risk minutes
- alert rate

During the week containing the July failure:

```text
Mean failure probability     17.60%
Maximum probability         100.00%
Alert rate                   18.52%
```

This provides an operational view of how model behavior changes over time.

---

# Streamlit Application

The final maintenance console contains five analytical sections.

```text
Equipment Health
Failure Intelligence
Model Performance
Monitoring
Business Impact
```

The dashboard allows users to switch between:

```text
Second failure warning window
Full production period
Latest 48 hours
```

This allows both failure investigation and routine monitoring.

---

# Technology Stack

## Data Engineering

```text
Python
DuckDB
SQL
PyArrow
Pandas
```

## Machine Learning

```text
Scikit-learn
Random Forest
Logistic Regression
XGBoost
LightGBM
Isolation Forest
```

## Explainability

```text
SHAP
```

## Statistical Monitoring

```text
SciPy
Population Stability Index
Kolmogorov-Smirnov Test
```

## Visualization

```text
Plotly
Matplotlib
Streamlit
```

---

# Project Structure

```text
Industrial Equipment Predictive Maintenance
│
├── app
│   └── maintenance console.py
│
├── data
│   ├── raw
│   │   ├── eurogate
│   │   └── metropt2
│   │
│   └── processed
│
├── images
│   ├── equipment health.png
│   ├── failure intelligence.png
│   ├── model performance.png
│   ├── monitoring.png
│   ├── business impact.png
│   ├── early_warning_24h shap importance.png
│   └── near_failure_6h shap importance.png
│
├── models
│   ├── isolation forest.pkl
│   │
│   └── failure prediction
│       ├── 6h
│       ├── 12h
│       └── 24h
│
├── reports
│   ├── model comparison.csv
│   ├── maintenance alert episodes.csv
│   ├── maintenance cost simulation.csv
│   ├── feature drift report.csv
│   ├── weekly model monitoring.csv
│   ├── selected model explanations.csv
│   ├── early_warning_24h shap importance.csv
│   └── near_failure_6h shap importance.csv
│
├── src
│   ├── profile metropt2.py
│   ├── build raw warehouse.py
│   ├── build feature layer.py
│   ├── build rolling features.py
│   ├── build anomaly detection.py
│   ├── train failure models.py
│   ├── validate failure windows.py
│   ├── build model explanations.py
│   ├── build maintenance decision layer.py
│   └── build model monitoring.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Pipeline Scripts

## `profile metropt2.py`

Profiles the raw MetroPT2 dataset.

Checks:

- schema
- sensor columns
- total row count
- timestamp availability
- missing values

---

## `build raw warehouse.py`

Creates the DuckDB warehouse.

Loads:

```text
7,116,940 raw readings
```

Also creates the documented failure-event table.

---

## `build feature layer.py`

Aggregates second-level telemetry into minute-level observations.

Creates:

- sensor aggregates
- failure-active labels
- 6-hour target
- 12-hour target
- 24-hour target
- recovery-period exclusions

---

## `build rolling features.py`

Creates backward-looking time-series features.

Includes:

- rolling averages
- rolling volatility
- trends
- z-scores
- deviations from baseline
- pressure relationships
- equipment activity
- data coverage

---

## `build anomaly detection.py`

Trains the Isolation Forest on healthy baseline operation.

Outputs:

- anomaly score
- anomaly percentile
- anomaly flag

---

## `train failure models.py`

Trains:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

for:

```text
6h
12h
24h
```

Uses chronological validation.

---

## `validate failure windows.py`

Checks actual telemetry availability before failures.

This validation prevented the project from incorrectly claiming that the Random Forest predicted failure six hours early.

It showed that only approximately 43 usable minutes were available immediately before the second failure for the six-hour target.

---

## `build model explanations.py`

Produces SHAP explanations for:

```text
24-hour Logistic Regression
6-hour Random Forest
```

Outputs both global feature importance and first-alert explanations.

---

## `build maintenance decision layer.py`

Combines:

```text
early-warning model
near-failure model
anomaly score
```

into:

- maintenance priority score
- risk level
- warning signal
- recommended maintenance action
- alert episodes
- maintenance cost simulation

---

## `build model monitoring.py`

Creates production-monitoring outputs.

Includes:

- PSI
- KS statistics
- weekly prediction monitoring
- alert-rate monitoring
- retraining-review status

---

# Installation

Clone the repository:

```bash
git clone YOUR-REPOSITORY-URL
```

Move into the project:

```bash
cd "Industrial Equipment Predictive Maintenance"
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

The raw MetroPT2 dataset is not included in this repository because the CSV is approximately 1.2 GB.

Download:

```text
MetroPT2.csv
```

from:

https://zenodo.org/records/7766691

Place it inside:

```text
data/raw/metropt2/MetroPT2.csv
```

---

# Rebuild the Project

Run the scripts in this order.

```bash
python "src/profile metropt2.py"
```

```bash
python "src/build raw warehouse.py"
```

```bash
python "src/build feature layer.py"
```

```bash
python "src/build rolling features.py"
```

```bash
python "src/build anomaly detection.py"
```

```bash
python "src/train failure models.py"
```

```bash
python "src/validate failure windows.py"
```

```bash
python "src/build model explanations.py"
```

```bash
python "src/build maintenance decision layer.py"
```

```bash
python "src/build model monitoring.py"
```

---

# Run the Dashboard

Start Streamlit:

```bash
streamlit run "app/maintenance console.py"
```

The dashboard will open in the browser.

---

# Generated Outputs

The pipeline produces:

## Processed datasets

```text
sensor minute features.parquet
rolling sensor features.parquet
anomaly sensor features.parquet
failure model predictions.parquet
maintenance decisions.parquet
```

## Models

```text
Isolation Forest
Logistic Regression
Random Forest
XGBoost
LightGBM
```

## Reports

```text
Model comparison
SHAP explanations
Maintenance alert episodes
Maintenance cost simulation
Feature drift report
Weekly model monitoring
```

---

# Important Modeling Limitations

This project deliberately avoids overstating model performance.

## Only Two Documented Failure Events

MetroPT2 contains only two documented failure periods.

This limits the ability to estimate how well the models would generalize across many independent equipment failures.

---

## 6-Hour Random Forest Result

The Random Forest produced:

```text
PR-AUC = 1.000
```

on the final six-hour target.

However, only 43 usable minute observations existed before the second failure in that target window.

The earliest usable observation occurred approximately:

```text
43 minutes before failure
```

Therefore the result is interpreted as:

```text
Near-failure detection
```

and not:

```text
Reliable six-hour advance prediction
```

---

## Financial Impact

The `$20,000` avoided-cost estimate is based on explicit scenario assumptions.

It should not be interpreted as an observed financial result from the MetroPT2 equipment.

---

## SHAP Interpretation

SHAP values explain model behavior.

They do not prove causal relationships between individual sensors and equipment failure.

Several time-series variables are correlated because they describe related operating conditions.

---

## Drift

Sensor drift may represent:

- changing equipment behavior
- operating regime changes
- environmental variation
- deterioration
- true failure-related behavior

A high PSI therefore triggers investigation rather than automatic retraining.

---

# Business Value

The project demonstrates how industrial analytics can connect machine learning with operational decision-making.

Instead of stopping at:

```text
Failure probability = 0.82
```

the system converts predictions into:

```text
Risk level
High

Main warning signal
Motor current instability

Recommended action
Inspect during next maintenance window

Estimated operational impact
Reduced unplanned downtime
```

That makes the analysis useful to:

- maintenance teams
- reliability engineers
- operations managers
- plant managers
- industrial analysts
- manufacturing leadership

---

# Skills Demonstrated

This project demonstrates experience with:

```text
Industrial Analytics
Manufacturing Analytics
IoT Analytics
Predictive Maintenance
Time-Series Analytics
Feature Engineering
Anomaly Detection
Machine Learning
Rare-Event Classification
Model Explainability
Model Monitoring
Data Engineering
SQL
DuckDB
Python
Business Analytics
Operational Decision Support
Cost Simulation
Dashboard Development
```

---

# Final Outcome

The project converts more than **7.1 million real industrial sensor readings** into an end-to-end predictive-maintenance system.

The final workflow:

```text
ingests raw IoT telemetry
        ↓
creates an analytical warehouse
        ↓
engineers time-series behavior
        ↓
detects equipment anomalies
        ↓
estimates failure risk
        ↓
explains model warnings
        ↓
assigns maintenance priority
        ↓
recommends operational action
        ↓
estimates downtime and maintenance impact
        ↓
monitors model and sensor drift
        ↓
delivers results through an interactive maintenance console
```

During chronological testing, the early-warning system generated a High-risk warning approximately:

```text
22.84 hours before the unseen July equipment failure
```

while the near-failure model detected the highly abnormal operating period immediately preceding the failure.

The project demonstrates how raw industrial telemetry can be converted into practical maintenance intelligence rather than remaining only as sensor data or isolated machine learning predictions.
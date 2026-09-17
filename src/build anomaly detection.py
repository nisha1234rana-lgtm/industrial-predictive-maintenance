from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import pickle
import time

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"

MODEL_PATH = PROJECT_ROOT / "models" / "isolation forest.pkl"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomaly sensor features.parquet"
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Isolation Forest anomaly detection")
print("=" * 70)

start = time.time()

con = duckdb.connect(str(DB_PATH))


# --------------------------------------------------
# Features used for anomaly detection
# --------------------------------------------------

FEATURES = [
    "oil_temp_long_trend",
    "oil_temp_std_1h",
    "oil_temp_zscore_6h",

    "motor_current_long_trend",
    "motor_current_std_1h",
    "motor_current_zscore_6h",

    "tp3_long_trend",
    "tp3_std_1h",
    "tp3_zscore_6h",

    "reservoir_std_1h",
    "reservoir_zscore_6h",

    "flow_std_1h",
    "flow_zscore_6h",

    "pressure_balance_gap",

    "compressor_activity_change",
]


# --------------------------------------------------
# Load modeling-quality observations
# --------------------------------------------------

print("\nLoading rolling features...")

feature_sql = ", ".join(FEATURES)

df = con.execute(
    f"""
    SELECT
        minute_timestamp,

        failure_active,
        failure_next_6h,
        failure_next_12h,
        failure_next_24h,
        post_failure_24h,
        eligible_for_model,

        {feature_sql}

    FROM sensor_rolling_features

    WHERE coverage_6h >= 0.80

    ORDER BY minute_timestamp
    """
).fetchdf()

print(f"Rows loaded: {len(df):,}")


# --------------------------------------------------
# Healthy training baseline
#
# Use only data before the first failure risk window.
# First failure starts June 4.
# Stop baseline June 3 to avoid learning
# pre-failure behaviour as normal.
# --------------------------------------------------

baseline_end = pd.Timestamp("2022-06-03 00:00:00")

training_mask = (
    (df["minute_timestamp"] < baseline_end)
    & (df["failure_active"] == 0)
    & (df["failure_next_24h"] == 0)
    & (df["post_failure_24h"] == 0)
)

train_df = df.loc[training_mask].copy()

print(f"Healthy baseline rows: {len(train_df):,}")
print(
    "Baseline period:",
    train_df["minute_timestamp"].min(),
    "to",
    train_df["minute_timestamp"].max(),
)


# --------------------------------------------------
# Missing / infinite value cleanup
# --------------------------------------------------

X_train = train_df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

medians = X_train.median()

X_train = X_train.fillna(medians)

X_all = df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

X_all = X_all.fillna(medians)


# --------------------------------------------------
# Standardize
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_all_scaled = scaler.transform(X_all)


# --------------------------------------------------
# Isolation Forest
# --------------------------------------------------

print("Training Isolation Forest...")

model = IsolationForest(
    n_estimators=400,
    contamination=0.01,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_scaled)


# --------------------------------------------------
# Scores
#
# score_samples:
# larger = more normal
#
# Reverse it so:
# larger anomaly_score = more abnormal
# --------------------------------------------------

train_scores = -model.score_samples(X_train_scaled)

all_scores = -model.score_samples(X_all_scaled)

df["anomaly_score"] = all_scores


# --------------------------------------------------
# Alert threshold
#
# Use 99th percentile of healthy baseline
# rather than arbitrary score.
# --------------------------------------------------

alert_threshold = np.quantile(
    train_scores,
    0.99
)

df["anomaly_flag"] = (
    df["anomaly_score"] >= alert_threshold
).astype(int)


# --------------------------------------------------
# Convert score to healthy-baseline percentile
#
# A value of 99 means the observation is more
# abnormal than about 99% of healthy baseline rows.
# --------------------------------------------------

sorted_train_scores = np.sort(train_scores)

df["anomaly_percentile"] = (
    np.searchsorted(
        sorted_train_scores,
        df["anomaly_score"],
        side="right",
    )
    / len(sorted_train_scores)
    * 100
)

df["anomaly_percentile"] = (
    df["anomaly_percentile"]
    .clip(0, 100)
    .round(2)
)


# --------------------------------------------------
# Save model objects
# --------------------------------------------------

model_bundle = {
    "model": model,
    "scaler": scaler,
    "features": FEATURES,
    "medians": medians.to_dict(),
    "alert_threshold": float(alert_threshold),
    "baseline_end": str(baseline_end),
}

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model_bundle, file)

print(f"\nModel saved: {MODEL_PATH}")


# --------------------------------------------------
# Write anomaly results back into DuckDB
# --------------------------------------------------

print("Writing anomaly results to DuckDB...")

anomaly_results = df[
    [
        "minute_timestamp",
        "anomaly_score",
        "anomaly_percentile",
        "anomaly_flag",
    ]
].copy()

con.register(
    "anomaly_results_df",
    anomaly_results,
)

con.execute(
    """
    CREATE OR REPLACE TABLE anomaly_scores AS

    SELECT *
    FROM anomaly_results_df
    """
)

con.execute(
    """
    CREATE OR REPLACE TABLE sensor_anomaly_features AS

    SELECT
        r.*,
        a.anomaly_score,
        a.anomaly_percentile,
        a.anomaly_flag

    FROM sensor_rolling_features r

    LEFT JOIN anomaly_scores a
        USING (minute_timestamp)

    ORDER BY minute_timestamp
    """
)


# --------------------------------------------------
# Export
# --------------------------------------------------

con.execute(
    f"""
    COPY sensor_anomaly_features
    TO '{OUTPUT_PATH.as_posix()}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """
)

print(f"Processed file saved: {OUTPUT_PATH}")


# --------------------------------------------------
# Validation
# --------------------------------------------------

print("\nANOMALY MODEL")
print(f"Features:              {len(FEATURES)}")
print(f"Training rows:         {len(train_df):,}")
print(f"Alert threshold:       {alert_threshold:.6f}")


# --------------------------------------------------
# Healthy baseline alert rate
# --------------------------------------------------

baseline_alert_rate = (
    df.loc[training_mask, "anomaly_flag"].mean() * 100
)

print(
    f"Baseline alert rate:   "
    f"{baseline_alert_rate:.2f}%"
)


# --------------------------------------------------
# Normal vs pre-failure
# --------------------------------------------------

eligible = df[
    (df["eligible_for_model"] == 1)
].copy()

comparison = (
    eligible
    .assign(
        period=np.where(
            eligible["failure_next_24h"] == 1,
            "Within 24h of failure",
            "Normal",
        )
    )
    .groupby("period")
    .agg(
        rows=("minute_timestamp", "size"),
        mean_anomaly_score=("anomaly_score", "mean"),
        median_percentile=("anomaly_percentile", "median"),
        anomaly_rate=("anomaly_flag", "mean"),
    )
    .reset_index()
)

comparison["anomaly_rate"] *= 100

print("\nNORMAL VS PRE-FAILURE ANOMALIES")

print(
    comparison.round(
        {
            "mean_anomaly_score": 4,
            "median_percentile": 2,
            "anomaly_rate": 2,
        }
    ).to_string(index=False)
)


# --------------------------------------------------
# 6 / 12 / 24 hour risk windows
# --------------------------------------------------

print("\nANOMALY RATE BY FAILURE HORIZON")

for horizon in [
    "failure_next_6h",
    "failure_next_12h",
    "failure_next_24h",
]:

    subset = eligible[
        eligible[horizon] == 1
    ]

    if len(subset) == 0:
        continue

    rate = (
        subset["anomaly_flag"].mean()
        * 100
    )

    median_percentile = (
        subset["anomaly_percentile"].median()
    )

    print(
        f"{horizon:<20} "
        f"rows={len(subset):>6,}   "
        f"alert_rate={rate:>6.2f}%   "
        f"median_percentile={median_percentile:>6.2f}"
    )


# --------------------------------------------------
# Individual failure events
# --------------------------------------------------

events = con.execute(
    """
    SELECT
        event_id,
        failure_type,
        start_time
    FROM failure_events
    ORDER BY event_id
    """
).fetchdf()

print("\n24 HOURS BEFORE EACH FAILURE")

for _, event in events.iterrows():

    event_start = pd.Timestamp(event["start_time"])

    event_window = df[
        (df["minute_timestamp"] >= event_start - pd.Timedelta(hours=24))
        & (df["minute_timestamp"] < event_start)
    ]

    if len(event_window) == 0:
        continue

    alert_rate = (
        event_window["anomaly_flag"].mean()
        * 100
    )

    median_percentile = (
        event_window["anomaly_percentile"].median()
    )

    max_percentile = (
        event_window["anomaly_percentile"].max()
    )

    print()
    print(
        f"Event {event['event_id']} - "
        f"{event['failure_type']}"
    )

    print(
        f"Rows:               {len(event_window):,}"
    )

    print(
        f"Anomaly alert rate: {alert_rate:.2f}%"
    )

    print(
        f"Median percentile:  {median_percentile:.2f}"
    )

    print(
        f"Maximum percentile: {max_percentile:.2f}"
    )


con.close()

elapsed = time.time() - start

print("\n" + "=" * 70)
print("ANOMALY DETECTION COMPLETE")
print(f"Runtime: {elapsed:.1f} seconds")
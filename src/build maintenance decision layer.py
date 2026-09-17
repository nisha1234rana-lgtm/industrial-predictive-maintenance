from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "maintenance decisions.parquet"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Maintenance decision and cost layer")
print("=" * 78)

con = duckdb.connect(str(DB_PATH))


# --------------------------------------------------
# Load selected model predictions
# --------------------------------------------------

early = con.execute(
    """
    SELECT
        minute_timestamp,
        failure_probability AS early_warning_score,
        alert_threshold AS early_warning_threshold,
        alert AS early_warning_alert

    FROM failure_predictions

    WHERE horizon = '24h'
      AND model = 'Logistic Regression'

    ORDER BY minute_timestamp
    """
).fetchdf()


near = con.execute(
    """
    SELECT
        minute_timestamp,
        failure_probability AS near_failure_score,
        alert_threshold AS near_failure_threshold,
        alert AS near_failure_alert

    FROM failure_predictions

    WHERE horizon = '6h'
      AND model = 'Random Forest'

    ORDER BY minute_timestamp
    """
).fetchdf()


sensor = con.execute(
    """
    SELECT
        minute_timestamp,
        anomaly_score,
        anomaly_percentile,
        anomaly_flag,

        oil_temperature_mean,
        oil_temp_long_trend,

        motor_current_mean,
        motor_current_avg_6h,
        motor_current_std_6h,

        tp3_mean,
        tp3_avg_6h,

        reservoirs_mean,
        reservoir_avg_6h,

        flowmeter_mean,
        flow_deviation_6h,

        failure_next_24h

    FROM sensor_anomaly_features

    WHERE minute_timestamp >= TIMESTAMP '2022-07-01'
      AND eligible_for_model = 1
      AND coverage_6h >= 0.80
      AND anomaly_score IS NOT NULL

    ORDER BY minute_timestamp
    """
).fetchdf()


failure_start = con.execute(
    """
    SELECT start_time
    FROM failure_events
    WHERE event_id = 2
    """
).fetchone()[0]

failure_start = pd.Timestamp(failure_start)


# --------------------------------------------------
# Combine layers
# --------------------------------------------------

df = sensor.merge(
    early,
    on="minute_timestamp",
    how="inner",
)

df = df.merge(
    near,
    on="minute_timestamp",
    how="inner",
)

df["minute_timestamp"] = pd.to_datetime(
    df["minute_timestamp"]
)


# --------------------------------------------------
# Normalize each model around its operating threshold
#
# Score = threshold -> normalized value around 0.5
# Twice threshold -> approaches 1
# --------------------------------------------------

def threshold_normalize(value, threshold):

    if threshold <= 0:
        return np.clip(value, 0, 1)

    ratio = value / threshold

    return np.clip(
        ratio / 2.0,
        0,
        1,
    )


df["early_component"] = [
    threshold_normalize(v, t)
    for v, t in zip(
        df["early_warning_score"],
        df["early_warning_threshold"],
    )
]

df["near_component"] = [
    threshold_normalize(v, t)
    for v, t in zip(
        df["near_failure_score"],
        df["near_failure_threshold"],
    )
]

df["anomaly_component"] = (
    df["anomaly_percentile"]
    / 100.0
)


# --------------------------------------------------
# Composite maintenance priority score
#
# Early warning         45%
# Near-failure signal   35%
# Unsupervised anomaly  20%
# --------------------------------------------------

df["maintenance_priority_score"] = (

    0.45 * df["early_component"]
    +
    0.35 * df["near_component"]
    +
    0.20 * df["anomaly_component"]

) * 100

df["maintenance_priority_score"] = (
    df["maintenance_priority_score"]
    .clip(0, 100)
    .round(1)
)


# --------------------------------------------------
# Risk level
# --------------------------------------------------

conditions = [

    # Immediate near-failure model alert
    (
        df["near_failure_alert"] == 1
    ),

    # Strong early warning plus abnormal telemetry
    (
        (df["early_warning_alert"] == 1)
        &
        (df["anomaly_percentile"] >= 95)
    ),

    # Early-warning model alert
    (
        df["early_warning_alert"] == 1
    ),

    # Sensor abnormality without supervised alert
    (
        (df["anomaly_percentile"] >= 95)
        |
        (df["maintenance_priority_score"] >= 45)
    ),
]

choices = [
    "Critical",
    "Critical",
    "High",
    "Medium",
]

df["risk_level"] = np.select(
    conditions,
    choices,
    default="Low",
)


# --------------------------------------------------
# Recommended maintenance action
# --------------------------------------------------

action_map = {

    "Critical":
        "Inspect immediately and prepare controlled maintenance shutdown",

    "High":
        "Inspect during the next available maintenance window",

    "Medium":
        "Increase monitoring and inspect if elevated risk persists",

    "Low":
        "Continue normal operation and routine monitoring",
}

df["recommended_action"] = (
    df["risk_level"]
    .map(action_map)
)


# --------------------------------------------------
# Identify primary warning signal
# --------------------------------------------------

def primary_signal(row):

    signals = {}

    signals[
        "Motor current instability"
    ] = abs(
        row["motor_current_mean"]
        - row["motor_current_avg_6h"]
    )

    signals[
        "Airflow deviation"
    ] = abs(
        row["flow_deviation_6h"]
    )

    signals[
        "Oil temperature trend"
    ] = abs(
        row["oil_temp_long_trend"]
    )

    signals[
        "TP3 pressure deviation"
    ] = abs(
        row["tp3_mean"]
        - row["tp3_avg_6h"]
    )

    signals[
        "Reservoir pressure deviation"
    ] = abs(
        row["reservoirs_mean"]
        - row["reservoir_avg_6h"]
    )

    return max(
        signals,
        key=signals.get,
    )


df["primary_warning_signal"] = (
    df.apply(
        primary_signal,
        axis=1,
    )
)


# --------------------------------------------------
# Find maintenance alert episodes
#
# Multiple consecutive alert minutes are ONE episode,
# not hundreds of separate maintenance calls.
# New episode starts after >60 minutes without
# High/Critical risk.
# --------------------------------------------------

alert_rows = df[
    df["risk_level"].isin(
        ["High", "Critical"]
    )
].copy()

alert_rows = alert_rows.sort_values(
    "minute_timestamp"
)

alert_rows["minutes_since_previous"] = (
    alert_rows["minute_timestamp"]
    .diff()
    .dt.total_seconds()
    / 60
)

alert_rows["new_episode"] = (
    alert_rows["minutes_since_previous"].isna()
    |
    (
        alert_rows[
            "minutes_since_previous"
        ] > 60
    )
).astype(int)

alert_rows["alert_episode_id"] = (
    alert_rows["new_episode"]
    .cumsum()
)


episodes = (
    alert_rows
    .groupby("alert_episode_id")
    .agg(
        episode_start=(
            "minute_timestamp",
            "min",
        ),

        episode_end=(
            "minute_timestamp",
            "max",
        ),

        max_priority_score=(
            "maintenance_priority_score",
            "max",
        ),

        peak_early_warning_score=(
            "early_warning_score",
            "max",
        ),

        peak_near_failure_score=(
            "near_failure_score",
            "max",
        ),

        peak_anomaly_percentile=(
            "anomaly_percentile",
            "max",
        ),

        observations=(
            "minute_timestamp",
            "size",
        ),
    )
    .reset_index()
)


# --------------------------------------------------
# Determine whether alert episode occurs in the
# 24 hours before the known test failure
# --------------------------------------------------

failure_window_start = (
    failure_start
    - pd.Timedelta(hours=24)
)

episodes["failure_related"] = (

    (episodes["episode_start"] < failure_start)
    &
    (episodes["episode_end"] >= failure_window_start)

).astype(int)


# --------------------------------------------------
# Cost simulation assumptions
#
# These are explicitly scenario assumptions.
# They are NOT costs supplied by MetroPT2.
# --------------------------------------------------

DOWNTIME_COST_PER_HOUR = 2500

REACTIVE_DOWNTIME_HOURS = 12
PREDICTIVE_DOWNTIME_HOURS = 4

EMERGENCY_REPAIR_COST = 12000
PLANNED_REPAIR_COST = 5000

INSPECTION_COST = 500


# --------------------------------------------------
# Reactive maintenance scenario
# --------------------------------------------------

reactive_downtime_cost = (
    DOWNTIME_COST_PER_HOUR
    * REACTIVE_DOWNTIME_HOURS
)

reactive_total_cost = (
    reactive_downtime_cost
    + EMERGENCY_REPAIR_COST
)


# --------------------------------------------------
# Predictive maintenance scenario
# --------------------------------------------------

predictive_downtime_cost = (
    DOWNTIME_COST_PER_HOUR
    * PREDICTIVE_DOWNTIME_HOURS
)

failure_alert_episodes = int(
    episodes["failure_related"].sum()
)

false_alert_episodes = int(
    (episodes["failure_related"] == 0).sum()
)

inspection_cost = (
    len(episodes)
    * INSPECTION_COST
)

predictive_total_cost = (
    predictive_downtime_cost
    + PLANNED_REPAIR_COST
    + inspection_cost
)

estimated_avoided_cost = (
    reactive_total_cost
    - predictive_total_cost
)


# --------------------------------------------------
# Downtime avoided
# --------------------------------------------------

downtime_hours_avoided = (
    REACTIVE_DOWNTIME_HOURS
    - PREDICTIVE_DOWNTIME_HOURS
)


# --------------------------------------------------
# First useful warning
# --------------------------------------------------

failure_related_alerts = alert_rows[
    (
        alert_rows["minute_timestamp"]
        >= failure_window_start
    )
    &
    (
        alert_rows["minute_timestamp"]
        < failure_start
    )
]

if len(failure_related_alerts) > 0:

    first_warning = (
        failure_related_alerts[
            "minute_timestamp"
        ].min()
    )

    warning_lead_hours = (
        failure_start
        - first_warning
    ).total_seconds() / 3600

else:

    first_warning = pd.NaT
    warning_lead_hours = np.nan


# --------------------------------------------------
# Cost report
# --------------------------------------------------

cost_report = pd.DataFrame(
    [
        {
            "scenario":
                "Reactive maintenance",

            "downtime_hours":
                REACTIVE_DOWNTIME_HOURS,

            "downtime_cost":
                reactive_downtime_cost,

            "repair_cost":
                EMERGENCY_REPAIR_COST,

            "inspection_cost":
                0,

            "total_cost":
                reactive_total_cost,
        },

        {
            "scenario":
                "Predictive maintenance",

            "downtime_hours":
                PREDICTIVE_DOWNTIME_HOURS,

            "downtime_cost":
                predictive_downtime_cost,

            "repair_cost":
                PLANNED_REPAIR_COST,

            "inspection_cost":
                inspection_cost,

            "total_cost":
                predictive_total_cost,
        },
    ]
)


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

df.to_parquet(
    OUTPUT_PATH,
    index=False,
)

episodes.to_csv(
    REPORT_DIR
    / "maintenance alert episodes.csv",
    index=False,
)

cost_report.to_csv(
    REPORT_DIR
    / "maintenance cost simulation.csv",
    index=False,
)


# --------------------------------------------------
# DuckDB tables
# --------------------------------------------------

con.register(
    "maintenance_df",
    df,
)

con.execute(
    """
    CREATE OR REPLACE TABLE maintenance_decisions AS

    SELECT *
    FROM maintenance_df
    """
)

con.register(
    "episodes_df",
    episodes,
)

con.execute(
    """
    CREATE OR REPLACE TABLE maintenance_alert_episodes AS

    SELECT *
    FROM episodes_df
    """
)

con.register(
    "cost_df",
    cost_report,
)

con.execute(
    """
    CREATE OR REPLACE TABLE maintenance_cost_simulation AS

    SELECT *
    FROM cost_df
    """
)

con.close()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nMAINTENANCE DECISION LAYER")
print("-" * 78)

print(
    f"Decision rows:              "
    f"{len(df):,}"
)

print(
    f"High/Critical episodes:     "
    f"{len(episodes):,}"
)

print(
    f"Failure-related episodes:   "
    f"{failure_alert_episodes:,}"
)

print(
    f"False alert episodes:       "
    f"{false_alert_episodes:,}"
)


print("\nRISK LEVEL DISTRIBUTION")
print(
    df["risk_level"]
    .value_counts()
    .to_string()
)


print("\nSECOND FAILURE WARNING")

if pd.notna(first_warning):

    print(
        f"First High/Critical warning: "
        f"{first_warning}"
    )

    print(
        f"Warning lead time:           "
        f"{warning_lead_hours:.2f} hours"
    )

else:

    print("No High/Critical warning detected.")


print("\nCOST SIMULATION")
print("-" * 78)

print(
    cost_report.to_string(
        index=False
    )
)

print()
print(
    f"Modeled downtime avoided:    "
    f"{downtime_hours_avoided:.1f} hours"
)

print(
    f"Modeled avoided cost:        "
    f"${estimated_avoided_cost:,.0f}"
)

print()
print(
    "NOTE: Cost values are scenario assumptions "
    "for business simulation, not observed MetroPT2 costs."
)

print("\n" + "=" * 78)
print("MAINTENANCE DECISION LAYER COMPLETE")
print(f"Output: {OUTPUT_PATH}")
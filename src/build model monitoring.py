from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Model and sensor drift monitoring")
print("=" * 78)

con = duckdb.connect(str(DB_PATH))


# --------------------------------------------------
# Features to monitor
# --------------------------------------------------

FEATURES = [
    "oil_temperature_mean",
    "oil_temp_long_trend",
    "oil_temp_std_1h",

    "motor_current_mean",
    "motor_current_avg_6h",
    "motor_current_std_6h",

    "tp3_mean",
    "tp3_avg_6h",
    "tp3_std_1h",

    "reservoirs_mean",
    "reservoir_avg_6h",

    "flowmeter_mean",
    "flow_deviation_6h",

    "anomaly_score",
    "anomaly_percentile",
]


# --------------------------------------------------
# Load full monitored dataset
# --------------------------------------------------

feature_sql = ", ".join(FEATURES)

df = con.execute(
    f"""
    SELECT
        minute_timestamp,
        failure_next_24h,
        eligible_for_model,
        coverage_6h,
        {feature_sql}

    FROM sensor_anomaly_features

    WHERE eligible_for_model = 1
      AND coverage_6h >= 0.80
      AND anomaly_score IS NOT NULL

    ORDER BY minute_timestamp
    """
).fetchdf()

df["minute_timestamp"] = pd.to_datetime(
    df["minute_timestamp"]
)


# --------------------------------------------------
# Production model predictions
# --------------------------------------------------

predictions = con.execute(
    """
    SELECT
        minute_timestamp,
        failure_probability,
        alert
    FROM failure_predictions
    WHERE horizon = '24h'
      AND model = 'Logistic Regression'
    ORDER BY minute_timestamp
    """
).fetchdf()

predictions["minute_timestamp"] = pd.to_datetime(
    predictions["minute_timestamp"]
)


# --------------------------------------------------
# Baseline
#
# Healthy operation before first failure risk period
# --------------------------------------------------

BASELINE_END = pd.Timestamp(
    "2022-06-03 00:00:00"
)

baseline = df[
    df["minute_timestamp"] < BASELINE_END
].copy()

production = df[
    df["minute_timestamp"] >= pd.Timestamp(
        "2022-07-01 00:00:00"
    )
].copy()

print(f"\nBaseline rows:   {len(baseline):,}")
print(f"Production rows: {len(production):,}")


# --------------------------------------------------
# PSI
# --------------------------------------------------

def calculate_psi(expected, actual, bins=10):

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Baseline quantile bins
    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    boundaries = np.unique(
        expected.quantile(
            quantiles
        ).to_numpy()
    )

    if len(boundaries) < 3:
        return 0.0

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    expected_bins = pd.cut(
        expected,
        bins=boundaries,
        include_lowest=True,
    )

    actual_bins = pd.cut(
        actual,
        bins=boundaries,
        include_lowest=True,
    )

    expected_pct = (
        expected_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    actual_pct = (
        actual_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
        .reindex(
            expected_pct.index,
            fill_value=0,
        )
    )

    epsilon = 0.0001

    expected_pct = expected_pct.clip(
        lower=epsilon
    )

    actual_pct = actual_pct.clip(
        lower=epsilon
    )

    psi = (
        (
            actual_pct
            - expected_pct
        )
        *
        np.log(
            actual_pct
            / expected_pct
        )
    ).sum()

    return float(psi)


# --------------------------------------------------
# Drift classification
# --------------------------------------------------

def drift_level(psi):

    if pd.isna(psi):
        return "Unknown"

    if psi < 0.10:
        return "Stable"

    if psi < 0.25:
        return "Moderate"

    return "High"


# --------------------------------------------------
# Feature drift
# --------------------------------------------------

drift_rows = []

for feature in FEATURES:

    base_values = baseline[
        feature
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    prod_values = production[
        feature
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    psi = calculate_psi(
        base_values,
        prod_values,
    )

    if (
        len(base_values) > 0
        and len(prod_values) > 0
    ):

        ks = ks_2samp(
            base_values,
            prod_values,
        )

        ks_statistic = float(
            ks.statistic
        )

        ks_pvalue = float(
            ks.pvalue
        )

    else:

        ks_statistic = np.nan
        ks_pvalue = np.nan

    drift_rows.append(
        {
            "feature": feature,

            "baseline_mean":
                base_values.mean(),

            "production_mean":
                prod_values.mean(),

            "baseline_std":
                base_values.std(),

            "production_std":
                prod_values.std(),

            "psi":
                psi,

            "drift_level":
                drift_level(psi),

            "ks_statistic":
                ks_statistic,

            "ks_pvalue":
                ks_pvalue,
        }
    )


drift = pd.DataFrame(
    drift_rows
)

drift = drift.sort_values(
    "psi",
    ascending=False,
)


# --------------------------------------------------
# Weekly operational monitoring
# --------------------------------------------------

monitor = production.merge(
    predictions,
    on="minute_timestamp",
    how="left",
)

monitor["week_start"] = (
    monitor["minute_timestamp"]
    .dt.to_period("W")
    .apply(
        lambda x: x.start_time
    )
)

weekly = (
    monitor
    .groupby(
        "week_start"
    )
    .agg(
        observations=(
            "minute_timestamp",
            "size",
        ),

        mean_failure_probability=(
            "failure_probability",
            "mean",
        ),

        max_failure_probability=(
            "failure_probability",
            "max",
        ),

        alert_rate=(
            "alert",
            "mean",
        ),

        mean_anomaly_score=(
            "anomaly_score",
            "mean",
        ),

        mean_anomaly_percentile=(
            "anomaly_percentile",
            "mean",
        ),

        true_risk_minutes=(
            "failure_next_24h",
            "sum",
        ),
    )
    .reset_index()
)

weekly["alert_rate_percent"] = (
    weekly["alert_rate"]
    * 100
)

weekly = weekly.drop(
    columns=["alert_rate"]
)


# --------------------------------------------------
# Monitoring status
# --------------------------------------------------

high_drift_features = int(
    (
        drift["drift_level"]
        == "High"
    ).sum()
)

moderate_drift_features = int(
    (
        drift["drift_level"]
        == "Moderate"
    ).sum()
)

stable_features = int(
    (
        drift["drift_level"]
        == "Stable"
    ).sum()
)


if high_drift_features >= 3:

    monitoring_status = (
        "Retraining review recommended"
    )

elif high_drift_features >= 1:

    monitoring_status = (
        "Investigate feature drift"
    )

elif moderate_drift_features >= 3:

    monitoring_status = (
        "Monitor closely"
    )

else:

    monitoring_status = (
        "Stable"
    )


# --------------------------------------------------
# Save reports
# --------------------------------------------------

drift.to_csv(
    REPORT_DIR
    / "feature drift report.csv",
    index=False,
)

weekly.to_csv(
    REPORT_DIR
    / "weekly model monitoring.csv",
    index=False,
)


# --------------------------------------------------
# Store in DuckDB
# --------------------------------------------------

con.register(
    "drift_df",
    drift,
)

con.execute(
    """
    CREATE OR REPLACE TABLE feature_drift_monitoring AS

    SELECT *
    FROM drift_df
    """
)

con.register(
    "weekly_df",
    weekly,
)

con.execute(
    """
    CREATE OR REPLACE TABLE weekly_model_monitoring AS

    SELECT *
    FROM weekly_df
    """
)

con.close()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nFEATURE DRIFT")
print("-" * 78)

print(
    drift[
        [
            "feature",
            "baseline_mean",
            "production_mean",
            "psi",
            "drift_level",
            "ks_statistic",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\nDRIFT SUMMARY")
print("-" * 78)

print(
    f"Stable features:       "
    f"{stable_features}"
)

print(
    f"Moderate drift:        "
    f"{moderate_drift_features}"
)

print(
    f"High drift:            "
    f"{high_drift_features}"
)

print(
    f"Monitoring status:     "
    f"{monitoring_status}"
)


print("\nWEEKLY MODEL MONITORING")
print("-" * 78)

print(
    weekly.round(
        {
            "mean_failure_probability": 4,
            "max_failure_probability": 4,
            "alert_rate_percent": 2,
            "mean_anomaly_score": 4,
            "mean_anomaly_percentile": 2,
        }
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("MODEL MONITORING COMPLETE")
print(
    f"Reports: "
    f"{REPORT_DIR}"
)
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"

MODEL_DIR = PROJECT_ROOT / "models" / "failure prediction"
REPORT_DIR = PROJECT_ROOT / "reports"

METRICS_PATH = REPORT_DIR / "model comparison.csv"
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "failure model predictions.parquet"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Supervised failure prediction")
print("=" * 75)

start = time.time()

con = duckdb.connect(str(DB_PATH))


# --------------------------------------------------
# Features
# --------------------------------------------------

FEATURES = [

    # Current operating condition
    "tp2_mean",
    "tp2_std",

    "tp3_mean",
    "tp3_std",

    "reservoirs_mean",
    "reservoirs_std",

    "oil_temperature_mean",
    "oil_temperature_std",

    "flowmeter_mean",
    "flowmeter_std",

    "motor_current_mean",
    "motor_current_std",

    "comp_active_rate",
    "lps_active_rate",
    "pressure_switch_active_rate",
    "oil_level_active_rate",

    # Oil temperature behaviour
    "oil_temp_avg_1h",
    "oil_temp_avg_6h",
    "oil_temp_std_1h",
    "oil_temp_std_6h",
    "oil_temp_short_trend",
    "oil_temp_long_trend",
    "oil_temp_deviation_6h",
    "oil_temp_zscore_6h",

    # Motor current behaviour
    "motor_current_avg_1h",
    "motor_current_avg_6h",
    "motor_current_std_1h",
    "motor_current_std_6h",
    "motor_current_short_trend",
    "motor_current_long_trend",
    "motor_current_zscore_6h",

    # Pressure behaviour
    "tp3_avg_1h",
    "tp3_avg_6h",
    "tp3_std_1h",
    "tp3_std_6h",
    "tp3_short_trend",
    "tp3_long_trend",
    "tp3_zscore_6h",

    "reservoir_avg_1h",
    "reservoir_avg_6h",
    "reservoir_std_1h",
    "reservoir_std_6h",
    "reservoir_deviation_6h",
    "reservoir_zscore_6h",

    "pressure_balance_gap",

    # Air flow
    "flow_avg_1h",
    "flow_avg_6h",
    "flow_std_1h",
    "flow_std_6h",
    "flow_deviation_6h",
    "flow_zscore_6h",

    # Compressor behaviour
    "compressor_activity_1h",
    "compressor_activity_6h",
    "compressor_activity_change",

    # Unsupervised warning layer
    "anomaly_score",
    "anomaly_percentile",
    "anomaly_flag",
]


TARGETS = {
    "6h": "failure_next_6h",
    "12h": "failure_next_12h",
    "24h": "failure_next_24h",
}


# --------------------------------------------------
# Load modeling dataset
# --------------------------------------------------

print("\nLoading modeling dataset...")

feature_sql = ", ".join(FEATURES)

df = con.execute(
    f"""
    SELECT
        minute_timestamp,
        failure_next_6h,
        failure_next_12h,
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

print(f"Rows available: {len(df):,}")
print(f"Features:       {len(FEATURES)}")


# --------------------------------------------------
# Strict chronological split
# --------------------------------------------------

TRAIN_END = pd.Timestamp("2022-06-15 00:00:00")
CALIBRATION_END = pd.Timestamp("2022-07-01 00:00:00")

train = df[
    df["minute_timestamp"] < TRAIN_END
].copy()

calibration = df[
    (df["minute_timestamp"] >= TRAIN_END)
    & (df["minute_timestamp"] < CALIBRATION_END)
].copy()

test = df[
    df["minute_timestamp"] >= CALIBRATION_END
].copy()

print("\nTIME SPLIT")
print(
    f"Training:     {train['minute_timestamp'].min()} "
    f"to {train['minute_timestamp'].max()} "
    f"({len(train):,} rows)"
)

print(
    f"Calibration:  {calibration['minute_timestamp'].min()} "
    f"to {calibration['minute_timestamp'].max()} "
    f"({len(calibration):,} rows)"
)

print(
    f"Test:         {test['minute_timestamp'].min()} "
    f"to {test['minute_timestamp'].max()} "
    f"({len(test):,} rows)"
)


# --------------------------------------------------
# Test failure
# --------------------------------------------------

failure_event = con.execute(
    """
    SELECT
        start_time
    FROM failure_events
    WHERE event_id = 2
    """
).fetchone()

SECOND_FAILURE_START = pd.Timestamp(failure_event[0])


# --------------------------------------------------
# Prepare common features
# --------------------------------------------------

X_train_raw = (
    train[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
)

medians = X_train_raw.median()

X_train = X_train_raw.fillna(medians)

X_calibration = (
    calibration[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(medians)
)

X_test = (
    test[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(medians)
)


# --------------------------------------------------
# Scaler for logistic regression
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_calibration_scaled = scaler.transform(X_calibration)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Storage
# --------------------------------------------------

metric_rows = []
prediction_frames = []


# --------------------------------------------------
# Train each horizon
# --------------------------------------------------

for horizon_name, target in TARGETS.items():

    print("\n" + "=" * 75)
    print(f"PREDICTION HORIZON: {horizon_name.upper()}")
    print("=" * 75)

    y_train = train[target].astype(int)
    y_calibration = calibration[target].astype(int)
    y_test = test[target].astype(int)

    train_positive = int(y_train.sum())
    test_positive = int(y_test.sum())

    train_negative = len(y_train) - train_positive

    print(f"Training positives: {train_positive:,}")
    print(f"Training negatives: {train_negative:,}")
    print(f"Test positives:     {test_positive:,}")

    if train_positive == 0:
        print("No positive examples. Skipping.")
        continue

    scale_pos_weight = (
        train_negative / train_positive
    )

    models = {

        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        ),
    }

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        # ------------------------------------------
        # Train
        # ------------------------------------------

        if model_name == "Logistic Regression":

            model.fit(
                X_train_scaled,
                y_train,
            )

            cal_probability = model.predict_proba(
                X_calibration_scaled
            )[:, 1]

            test_probability = model.predict_proba(
                X_test_scaled
            )[:, 1]

        else:

            model.fit(
                X_train,
                y_train,
            )

            cal_probability = model.predict_proba(
                X_calibration
            )[:, 1]

            test_probability = model.predict_proba(
                X_test
            )[:, 1]

        # ------------------------------------------
        # Calibration threshold
        #
        # Threshold chosen using NORMAL calibration
        # period only.
        #
        # Approximately 1% alert budget.
        # ------------------------------------------

        calibration_normal = (
            y_calibration == 0
        )

        normal_probabilities = (
            cal_probability[calibration_normal]
        )

        threshold = float(
            np.quantile(
                normal_probabilities,
                0.99,
            )
        )

        test_prediction = (
            test_probability >= threshold
        ).astype(int)

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        pr_auc = average_precision_score(
            y_test,
            test_probability,
        )

        roc_auc = roc_auc_score(
            y_test,
            test_probability,
        )

        precision = precision_score(
            y_test,
            test_prediction,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            test_prediction,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            test_prediction,
            zero_division=0,
        )

        alert_rate = (
            test_prediction.mean() * 100
        )

        test_negative_mask = (
            y_test.to_numpy() == 0
        )

        false_alert_rate = (
            test_prediction[
                test_negative_mask
            ].mean()
            * 100
        )

        # ------------------------------------------
        # Failure-window alert coverage
        # ------------------------------------------

        positive_mask = (
            y_test.to_numpy() == 1
        )

        if positive_mask.sum() > 0:

            failure_window_alert_rate = (
                test_prediction[
                    positive_mask
                ].mean()
                * 100
            )

        else:

            failure_window_alert_rate = np.nan

        # ------------------------------------------
        # Earliest warning before failure
        # ------------------------------------------

        prediction_temp = pd.DataFrame(
            {
                "minute_timestamp":
                    test["minute_timestamp"].to_numpy(),

                "target":
                    y_test.to_numpy(),

                "probability":
                    test_probability,

                "alert":
                    test_prediction,
            }
        )

        hours = int(
            horizon_name.replace("h", "")
        )

        event_window = prediction_temp[
            (
                prediction_temp["minute_timestamp"]
                >= SECOND_FAILURE_START
                - pd.Timedelta(hours=hours)
            )
            &
            (
                prediction_temp["minute_timestamp"]
                < SECOND_FAILURE_START
            )
        ]

        event_alerts = event_window[
            event_window["alert"] == 1
        ]

        if len(event_alerts) > 0:

            first_alert = (
                event_alerts[
                    "minute_timestamp"
                ].min()
            )

            warning_lead_hours = (
                SECOND_FAILURE_START
                - first_alert
            ).total_seconds() / 3600

        else:

            warning_lead_hours = np.nan

        # ------------------------------------------
        # Output metrics
        # ------------------------------------------

        metric_rows.append(
            {
                "horizon": horizon_name,
                "model": model_name,
                "train_rows": len(train),
                "train_positive": train_positive,
                "test_rows": len(test),
                "test_positive": test_positive,
                "threshold": threshold,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "alert_rate_percent": alert_rate,
                "false_alert_rate_percent":
                    false_alert_rate,
                "failure_window_alert_percent":
                    failure_window_alert_rate,
                "first_warning_hours":
                    warning_lead_hours,
            }
        )

        # ------------------------------------------
        # Predictions
        # ------------------------------------------

        prediction_output = pd.DataFrame(
            {
                "minute_timestamp":
                    test["minute_timestamp"].to_numpy(),

                "horizon":
                    horizon_name,

                "model":
                    model_name,

                "actual":
                    y_test.to_numpy(),

                "failure_probability":
                    test_probability,

                "alert_threshold":
                    threshold,

                "alert":
                    test_prediction,
            }
        )

        prediction_frames.append(
            prediction_output
        )

        # ------------------------------------------
        # Save model
        # ------------------------------------------

        safe_model_name = (
            model_name.lower()
            .replace(" ", "-")
        )

        horizon_dir = (
            MODEL_DIR / horizon_name
        )

        horizon_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_bundle = {
            "model": model,
            "model_name": model_name,
            "horizon": horizon_name,
            "target": target,
            "features": FEATURES,
            "medians": medians.to_dict(),
            "threshold": threshold,
            "train_end": str(TRAIN_END),
            "calibration_end":
                str(CALIBRATION_END),
        }

        if model_name == "Logistic Regression":
            model_bundle["scaler"] = scaler

        model_path = (
            horizon_dir
            / f"{safe_model_name}.pkl"
        )

        joblib.dump(
            model_bundle,
            model_path,
        )

        print(
            f"PR-AUC={pr_auc:.4f} | "
            f"ROC-AUC={roc_auc:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

        print(
            f"Failure-window alerts="
            f"{failure_window_alert_rate:.2f}% | "
            f"False alerts="
            f"{false_alert_rate:.2f}%"
        )

        if np.isnan(warning_lead_hours):

            print(
                "First warning: none inside "
                f"{horizon_name} window"
            )

        else:

            print(
                f"First warning: "
                f"{warning_lead_hours:.2f} "
                f"hours before failure"
            )


# --------------------------------------------------
# Metrics table
# --------------------------------------------------

metrics = pd.DataFrame(metric_rows)

metrics = metrics.sort_values(
    ["horizon", "pr_auc"],
    ascending=[True, False],
)

metrics.to_csv(
    METRICS_PATH,
    index=False,
)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

display_columns = [
    "horizon",
    "model",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "false_alert_rate_percent",
    "failure_window_alert_percent",
    "first_warning_hours",
]

print(
    metrics[
        display_columns
    ].round(4).to_string(index=False)
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

all_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

all_predictions.to_parquet(
    PREDICTIONS_PATH,
    index=False,
)


# --------------------------------------------------
# Store in DuckDB
# --------------------------------------------------

con.register(
    "model_metrics_df",
    metrics,
)

con.execute(
    """
    CREATE OR REPLACE TABLE model_comparison AS

    SELECT *
    FROM model_metrics_df
    """
)

con.register(
    "model_predictions_df",
    all_predictions,
)

con.execute(
    """
    CREATE OR REPLACE TABLE failure_predictions AS

    SELECT *
    FROM model_predictions_df
    """
)

con.close()


elapsed = time.time() - start

print("\n" + "=" * 75)
print("FAILURE MODELING COMPLETE")
print(f"Metrics:     {METRICS_PATH}")
print(f"Predictions: {PREDICTIONS_PATH}")
print(f"Models:      {MODEL_DIR}")
print(f"Runtime:     {elapsed:.1f} seconds")
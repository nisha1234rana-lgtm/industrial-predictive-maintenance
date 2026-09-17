from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "failure prediction"
)

REPORT_DIR = PROJECT_ROOT / "reports"
IMAGE_DIR = PROJECT_ROOT / "images"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Model explainability")
print("=" * 75)


# --------------------------------------------------
# Selected production-style models
# --------------------------------------------------

MODELS = {
    "early_warning_24h": {
        "path":
            MODEL_ROOT
            / "24h"
            / "logistic-regression.pkl",

        "type": "logistic",

        "description":
            "24-hour early warning model",
    },

    "near_failure_6h": {
        "path":
            MODEL_ROOT
            / "6h"
            / "random-forest.pkl",

        "type": "tree",

        "description":
            "Near-failure Random Forest detector",
    },
}


# --------------------------------------------------
# Load data
# --------------------------------------------------

con = duckdb.connect(str(DB_PATH))

second_failure = con.execute(
    """
    SELECT start_time
    FROM failure_events
    WHERE event_id = 2
    """
).fetchone()[0]

second_failure = pd.Timestamp(second_failure)

df = con.execute(
    """
    SELECT *
    FROM sensor_anomaly_features
    WHERE eligible_for_model = 1
      AND coverage_6h >= 0.80
      AND anomaly_score IS NOT NULL
      AND minute_timestamp >= TIMESTAMP '2022-07-01'
    ORDER BY minute_timestamp
    """
).fetchdf()

con.close()

df["minute_timestamp"] = pd.to_datetime(
    df["minute_timestamp"]
)

print(f"\nTest rows loaded: {len(df):,}")
print(f"Second failure:   {second_failure}")


# --------------------------------------------------
# Run explanations
# --------------------------------------------------

summary_rows = []


for model_key, config in MODELS.items():

    print("\n" + "=" * 75)
    print(config["description"])
    print("=" * 75)

    bundle = joblib.load(config["path"])

    model = bundle["model"]
    features = bundle["features"]
    medians = bundle["medians"]
    threshold = bundle["threshold"]

    X = (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )

    # ----------------------------------------------
    # Prediction probabilities
    # ----------------------------------------------

    if config["type"] == "logistic":

        scaler = bundle["scaler"]

        X_model = scaler.transform(X)

        probabilities = model.predict_proba(
            X_model
        )[:, 1]

    else:

        X_model = X

        probabilities = model.predict_proba(
            X_model
        )[:, 1]

    working = df[
        ["minute_timestamp"]
    ].copy()

    working["probability"] = probabilities

    working["alert"] = (
        working["probability"]
        >= threshold
    ).astype(int)

    # ----------------------------------------------
    # Select rows around the second failure
    # ----------------------------------------------

    if model_key == "early_warning_24h":

        window_start = (
            second_failure
            - pd.Timedelta(hours=24)
        )

    else:

        # Since usable data only begins about
        # 43 minutes before the failure for this
        # nominal 6-hour target, use actual available
        # observations rather than claiming 6 hours.
        window_start = (
            second_failure
            - pd.Timedelta(hours=1)
        )

    window_mask = (
        (df["minute_timestamp"] >= window_start)
        &
        (df["minute_timestamp"] < second_failure)
    )

    X_window = X.loc[window_mask].copy()

    working_window = (
        working.loc[window_mask].copy()
    )

    print(
        f"Explainability window rows: "
        f"{len(X_window):,}"
    )

    if len(X_window) == 0:
        continue

    # ----------------------------------------------
    # SHAP
    # ----------------------------------------------

    if config["type"] == "logistic":

        scaler = bundle["scaler"]

        X_window_scaled = scaler.transform(
            X_window
        )

        explainer = shap.LinearExplainer(
            model,
            X_model,
        )

        shap_values = explainer(
            X_window_scaled
        )

        shap_matrix = shap_values.values

    else:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer(
            X_window
        )

        values = shap_values.values

        # RandomForest binary classification may
        # return [rows, features, classes]
        if values.ndim == 3:
            shap_matrix = values[:, :, 1]
        else:
            shap_matrix = values

    # ----------------------------------------------
    # Global importance inside failure window
    # ----------------------------------------------

    mean_abs_shap = np.abs(
        shap_matrix
    ).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance_path = (
        REPORT_DIR
        / f"{model_key} shap importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print("\nTOP SHAP FEATURES")

    print(
        importance.head(12)
        .round(5)
        .to_string(index=False)
    )

    # ----------------------------------------------
    # SHAP bar chart
    # ----------------------------------------------

    top = (
        importance.head(12)
        .sort_values("mean_abs_shap")
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        top["feature"],
        top["mean_abs_shap"],
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel("")

    plt.title(
        config["description"]
        + "\nMain failure-risk drivers"
    )

    plt.tight_layout()

    plot_path = (
        IMAGE_DIR
        / f"{model_key} shap importance.png"
    )

    plt.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    # ----------------------------------------------
    # First alert explanation
    # ----------------------------------------------

    alerts = working_window[
        working_window["alert"] == 1
    ]

    if len(alerts) > 0:

        first_alert_timestamp = (
            alerts["minute_timestamp"].min()
        )

        position = (
            working_window
            .reset_index(drop=True)[
                "minute_timestamp"
            ]
            .eq(first_alert_timestamp)
            .idxmax()
        )

        row_shap = shap_matrix[position]

        actual_values = (
            X_window
            .reset_index(drop=True)
            .iloc[position]
        )

        explanation = pd.DataFrame(
            {
                "feature": features,
                "feature_value": [
                    actual_values[f]
                    for f in features
                ],
                "shap_value": row_shap,
                "absolute_impact":
                    np.abs(row_shap),
            }
        ).sort_values(
            "absolute_impact",
            ascending=False,
        )

        first_alert_probability = float(
            alerts.loc[
                alerts["minute_timestamp"]
                == first_alert_timestamp,
                "probability",
            ].iloc[0]
        )

        lead_hours = (
            second_failure
            - first_alert_timestamp
        ).total_seconds() / 3600

        explanation["alert_timestamp"] = (
            first_alert_timestamp
        )

        explanation[
            "failure_probability"
        ] = first_alert_probability

        explanation[
            "hours_before_failure"
        ] = lead_hours

        explanation_path = (
            REPORT_DIR
            / f"{model_key} first alert explanation.csv"
        )

        explanation.to_csv(
            explanation_path,
            index=False,
        )

        print("\nFIRST ALERT")

        print(
            f"Timestamp:           "
            f"{first_alert_timestamp}"
        )

        print(
            f"Hours before failure:"
            f" {lead_hours:.2f}"
        )

        print(
            f"Model probability:   "
            f"{first_alert_probability:.4f}"
        )

        print(
            f"Alert threshold:      "
            f"{threshold:.4f}"
        )

        print("\nTOP FIRST-ALERT DRIVERS")

        print(
            explanation[
                [
                    "feature",
                    "feature_value",
                    "shap_value",
                ]
            ]
            .head(10)
            .round(5)
            .to_string(index=False)
        )

        summary_rows.append(
            {
                "model":
                    config["description"],

                "first_alert_timestamp":
                    first_alert_timestamp,

                "hours_before_failure":
                    lead_hours,

                "failure_probability":
                    first_alert_probability,

                "threshold":
                    threshold,

                "top_driver_1":
                    explanation.iloc[0][
                        "feature"
                    ],

                "top_driver_2":
                    explanation.iloc[1][
                        "feature"
                    ],

                "top_driver_3":
                    explanation.iloc[2][
                        "feature"
                    ],
            }
        )

    else:

        print(
            "\nNo alert generated "
            "inside selected window."
        )


# --------------------------------------------------
# Save summary
# --------------------------------------------------

summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    REPORT_DIR
    / "selected model explanations.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print("\n" + "=" * 75)
print("MODEL EXPLANATIONS COMPLETE")
print(f"Summary: {summary_path}")
print(f"Charts:  {IMAGE_DIR}")
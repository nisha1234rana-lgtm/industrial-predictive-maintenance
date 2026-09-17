from pathlib import Path
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"

con = duckdb.connect(str(DB_PATH))

print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Failure-window validation")
print("=" * 78)

# --------------------------------------------------
# Failure events
# --------------------------------------------------

events = con.execute(
    """
    SELECT
        event_id,
        failure_type,
        start_time,
        end_time
    FROM failure_events
    ORDER BY event_id
    """
).fetchdf()

# --------------------------------------------------
# Modeling observations
# Must match the conditions used during model training.
# --------------------------------------------------

model_rows = con.execute(
    """
    SELECT
        minute_timestamp,
        failure_next_6h,
        failure_next_12h,
        failure_next_24h
    FROM sensor_anomaly_features
    WHERE eligible_for_model = 1
      AND coverage_6h >= 0.80
      AND anomaly_score IS NOT NULL
    ORDER BY minute_timestamp
    """
).fetchdf()

model_rows["minute_timestamp"] = pd.to_datetime(
    model_rows["minute_timestamp"]
)

# --------------------------------------------------
# Coverage around each failure
# --------------------------------------------------

print("\nOBSERVED DATA INSIDE PRE-FAILURE WINDOWS")
print("-" * 78)

for _, event in events.iterrows():

    failure_start = pd.Timestamp(event["start_time"])

    print()
    print(
        f"Event {event['event_id']} - "
        f"{event['failure_type']}"
    )
    print(f"Failure starts: {failure_start}")

    for hours in [6, 12, 24]:

        window_start = (
            failure_start
            - pd.Timedelta(hours=hours)
        )

        window = model_rows[
            (
                model_rows["minute_timestamp"]
                >= window_start
            )
            &
            (
                model_rows["minute_timestamp"]
                < failure_start
            )
        ].copy()

        expected_minutes = hours * 60
        observed_minutes = len(window)

        coverage = (
            observed_minutes
            / expected_minutes
            * 100
        )

        if len(window) > 0:

            first_observed = (
                window["minute_timestamp"].min()
            )

            last_observed = (
                window["minute_timestamp"].max()
            )

            sorted_times = (
                window["minute_timestamp"]
                .sort_values()
            )

            gaps = (
                sorted_times.diff()
                .dropna()
            )

            if len(gaps) > 0:
                largest_gap_minutes = (
                    gaps.max().total_seconds()
                    / 60
                )
            else:
                largest_gap_minutes = 0

            earliest_observed_lead = (
                failure_start
                - first_observed
            ).total_seconds() / 3600

        else:

            first_observed = None
            last_observed = None
            largest_gap_minutes = None
            earliest_observed_lead = None

        print()
        print(f"  {hours}h window")
        print(
            f"    Expected minutes:          "
            f"{expected_minutes:,}"
        )
        print(
            f"    Observed modeling minutes: "
            f"{observed_minutes:,}"
        )
        print(
            f"    Window coverage:           "
            f"{coverage:.2f}%"
        )

        if first_observed is not None:

            print(
                f"    First usable observation:  "
                f"{first_observed}"
            )

            print(
                f"    Last usable observation:   "
                f"{last_observed}"
            )

            print(
                f"    Earliest usable lead:      "
                f"{earliest_observed_lead:.2f} h"
            )

            print(
                f"    Largest telemetry gap:     "
                f"{largest_gap_minutes:.1f} min"
            )

# --------------------------------------------------
# Test-set model prediction inspection
# --------------------------------------------------

predictions = con.execute(
    """
    SELECT *
    FROM failure_predictions
    ORDER BY minute_timestamp
    """
).fetchdf()

predictions["minute_timestamp"] = pd.to_datetime(
    predictions["minute_timestamp"]
)

second_failure = events[
    events["event_id"] == 2
].iloc[0]

failure_start = pd.Timestamp(
    second_failure["start_time"]
)

print("\n" + "=" * 78)
print("SECOND FAILURE MODEL BEHAVIOUR")
print("=" * 78)

for horizon in ["6h", "12h", "24h"]:

    hours = int(
        horizon.replace("h", "")
    )

    start = (
        failure_start
        - pd.Timedelta(hours=hours)
    )

    horizon_predictions = predictions[
        (
            predictions["horizon"] == horizon
        )
        &
        (
            predictions["minute_timestamp"]
            >= start
        )
        &
        (
            predictions["minute_timestamp"]
            < failure_start
        )
    ].copy()

    print()
    print(f"{horizon.upper()} WINDOW")
    print("-" * 78)

    for model_name in sorted(
        horizon_predictions["model"].unique()
    ):

        subset = horizon_predictions[
            horizon_predictions["model"]
            == model_name
        ].copy()

        if len(subset) == 0:
            continue

        alerts = subset[
            subset["alert"] == 1
        ]

        if len(alerts) > 0:

            first_alert = (
                alerts["minute_timestamp"].min()
            )

            lead_hours = (
                failure_start
                - first_alert
            ).total_seconds() / 3600

        else:

            first_alert = None
            lead_hours = None

        print()
        print(model_name)
        print(
            f"  Observations:       "
            f"{len(subset):,}"
        )

        print(
            f"  Alerts:             "
            f"{int(subset['alert'].sum()):,}"
        )

        print(
            f"  Alert rate:         "
            f"{subset['alert'].mean() * 100:.2f}%"
        )

        print(
            f"  Mean probability:   "
            f"{subset['failure_probability'].mean():.4f}"
        )

        print(
            f"  Median probability: "
            f"{subset['failure_probability'].median():.4f}"
        )

        print(
            f"  Maximum probability:"
            f" {subset['failure_probability'].max():.4f}"
        )

        print(
            f"  Threshold:          "
            f"{subset['alert_threshold'].iloc[0]:.4f}"
        )

        if lead_hours is None:

            print(
                "  First warning:       None"
            )

        else:

            print(
                f"  First warning:       "
                f"{lead_hours:.2f} hours "
                f"before failure"
            )

# --------------------------------------------------
# Positive-row time distribution for second failure
# --------------------------------------------------

print("\n" + "=" * 78)
print("POSITIVE LABEL DISTRIBUTION BEFORE SECOND FAILURE")
print("=" * 78)

for target in [
    "failure_next_6h",
    "failure_next_12h",
    "failure_next_24h",
]:

    positive = model_rows[
        (
            model_rows[target] == 1
        )
        &
        (
            model_rows["minute_timestamp"]
            < failure_start
        )
        &
        (
            model_rows["minute_timestamp"]
            >= failure_start
            - pd.Timedelta(hours=24)
        )
    ].copy()

    if len(positive) == 0:
        continue

    earliest = positive[
        "minute_timestamp"
    ].min()

    latest = positive[
        "minute_timestamp"
    ].max()

    earliest_lead = (
        failure_start - earliest
    ).total_seconds() / 3600

    latest_lead = (
        failure_start - latest
    ).total_seconds() / 3600

    print()
    print(target)

    print(
        f"  Positive rows:      "
        f"{len(positive):,}"
    )

    print(
        f"  Earliest positive:  "
        f"{earliest}"
    )

    print(
        f"  Earliest lead:      "
        f"{earliest_lead:.2f} h"
    )

    print(
        f"  Latest positive:    "
        f"{latest}"
    )

    print(
        f"  Latest lead:        "
        f"{latest_lead:.2f} h"
    )

con.close()

print("\n" + "=" * 78)
print("VALIDATION COMPLETE")
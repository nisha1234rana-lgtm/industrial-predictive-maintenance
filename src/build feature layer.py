from pathlib import Path
import duckdb
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sensor minute features.parquet"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Building minute-level feature layer")
print("=" * 70)

start = time.time()

con = duckdb.connect(str(DB_PATH))

con.execute("SET preserve_insertion_order = false")

# --------------------------------------------------
# 1. Aggregate 1-second telemetry to 1-minute windows
# --------------------------------------------------

print("\nCreating minute-level sensor table...")

con.execute(
    """
    CREATE OR REPLACE TABLE sensor_minute AS

    SELECT
        DATE_TRUNC('minute', timestamp) AS minute_timestamp,

        COUNT(*) AS readings,

        AVG(tp2) AS tp2_mean,
        MIN(tp2) AS tp2_min,
        MAX(tp2) AS tp2_max,
        STDDEV_POP(tp2) AS tp2_std,

        AVG(tp3) AS tp3_mean,
        MIN(tp3) AS tp3_min,
        MAX(tp3) AS tp3_max,
        STDDEV_POP(tp3) AS tp3_std,

        AVG(h1) AS h1_mean,
        STDDEV_POP(h1) AS h1_std,

        AVG(dv_pressure) AS dv_pressure_mean,
        STDDEV_POP(dv_pressure) AS dv_pressure_std,

        AVG(reservoirs) AS reservoirs_mean,
        MIN(reservoirs) AS reservoirs_min,
        MAX(reservoirs) AS reservoirs_max,
        STDDEV_POP(reservoirs) AS reservoirs_std,

        AVG(oil_temperature) AS oil_temperature_mean,
        MIN(oil_temperature) AS oil_temperature_min,
        MAX(oil_temperature) AS oil_temperature_max,
        STDDEV_POP(oil_temperature) AS oil_temperature_std,

        AVG(flowmeter) AS flowmeter_mean,
        MAX(flowmeter) AS flowmeter_max,
        STDDEV_POP(flowmeter) AS flowmeter_std,

        AVG(motor_current) AS motor_current_mean,
        MAX(motor_current) AS motor_current_max,
        STDDEV_POP(motor_current) AS motor_current_std,

        AVG(comp) AS comp_active_rate,
        AVG(dv_electric) AS dv_electric_active_rate,
        AVG(towers) AS towers_active_rate,
        AVG(mpg) AS mpg_active_rate,
        AVG(lps) AS lps_active_rate,
        AVG(pressure_switch) AS pressure_switch_active_rate,
        AVG(oil_level) AS oil_level_active_rate,
        AVG(caudal_impulses) AS caudal_impulses_active_rate,

        AVG(gps_speed) AS gps_speed_mean,
        MAX(gps_speed) AS gps_speed_max,
        AVG(gps_quality) AS gps_quality_mean

    FROM sensor_readings_raw

    GROUP BY 1

    ORDER BY 1
    """
)

# --------------------------------------------------
# 2. Create failure-window labels
# --------------------------------------------------

print("Creating failure prediction labels...")

con.execute(
    """
    CREATE OR REPLACE TABLE failure_labeled_minutes AS

    SELECT
        m.*,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM failure_events f
                WHERE m.minute_timestamp
                      BETWEEN f.start_time AND f.end_time
            )
            THEN 1
            ELSE 0
        END AS failure_active,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM failure_events f
                WHERE f.start_time > m.minute_timestamp
                  AND f.start_time <= m.minute_timestamp
                                      + INTERVAL '6 hours'
            )
            THEN 1
            ELSE 0
        END AS failure_next_6h,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM failure_events f
                WHERE f.start_time > m.minute_timestamp
                  AND f.start_time <= m.minute_timestamp
                                      + INTERVAL '12 hours'
            )
            THEN 1
            ELSE 0
        END AS failure_next_12h,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM failure_events f
                WHERE f.start_time > m.minute_timestamp
                  AND f.start_time <= m.minute_timestamp
                                      + INTERVAL '24 hours'
            )
            THEN 1
            ELSE 0
        END AS failure_next_24h,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM failure_events f
                WHERE m.minute_timestamp > f.end_time
                  AND m.minute_timestamp <= f.end_time
                                      + INTERVAL '24 hours'
            )
            THEN 1
            ELSE 0
        END AS post_failure_24h,

        (
            SELECT MIN(
                EXTRACT(
                    EPOCH FROM (
                        f.start_time - m.minute_timestamp
                    )
                ) / 3600.0
            )
            FROM failure_events f
            WHERE f.start_time > m.minute_timestamp
        ) AS hours_to_next_failure

    FROM sensor_minute m

    ORDER BY m.minute_timestamp
    """
)

# --------------------------------------------------
# 3. Final modeling eligibility
# --------------------------------------------------

print("Creating modeling-ready labels...")

con.execute(
    """
    CREATE OR REPLACE TABLE sensor_failure_features AS

    SELECT
        *,

        CASE
            WHEN failure_active = 0
             AND post_failure_24h = 0
            THEN 1
            ELSE 0
        END AS eligible_for_model

    FROM failure_labeled_minutes
    """
)

# --------------------------------------------------
# 4. Export processed feature layer
# --------------------------------------------------

print("Exporting processed Parquet file...")

con.execute(
    f"""
    COPY sensor_failure_features
    TO '{OUTPUT_PATH.as_posix()}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """
)

# --------------------------------------------------
# 5. Validation
# --------------------------------------------------

summary = con.execute(
    """
    SELECT
        COUNT(*) AS minute_rows,
        MIN(minute_timestamp) AS first_minute,
        MAX(minute_timestamp) AS last_minute,
        ROUND(AVG(readings), 2) AS avg_readings_per_minute,
        MIN(readings) AS min_readings,
        MAX(readings) AS max_readings
    FROM sensor_failure_features
    """
).fetchone()

print("\nMINUTE DATASET")
print(f"Rows:                     {summary[0]:,}")
print(f"First minute:             {summary[1]}")
print(f"Last minute:              {summary[2]}")
print(f"Average readings/minute:  {summary[3]}")
print(f"Minimum readings/minute:  {summary[4]}")
print(f"Maximum readings/minute:  {summary[5]}")

labels = con.execute(
    """
    SELECT
        SUM(failure_active) AS active_failure_minutes,
        SUM(failure_next_6h) AS six_hour_positive_minutes,
        SUM(failure_next_12h) AS twelve_hour_positive_minutes,
        SUM(failure_next_24h) AS twenty_four_hour_positive_minutes,
        SUM(post_failure_24h) AS excluded_recovery_minutes,
        SUM(eligible_for_model) AS eligible_model_minutes
    FROM sensor_failure_features
    """
).fetchone()

print("\nFAILURE LABEL DISTRIBUTION")
print(f"Active failure minutes:       {labels[0]:,}")
print(f"Failure next 6h:              {labels[1]:,}")
print(f"Failure next 12h:             {labels[2]:,}")
print(f"Failure next 24h:             {labels[3]:,}")
print(f"Post-failure excluded:        {labels[4]:,}")
print(f"Eligible modeling minutes:    {labels[5]:,}")

# --------------------------------------------------
# Failure windows individually
# --------------------------------------------------

windows = con.execute(
    """
    SELECT
        event_id,
        failure_type,
        start_time,
        end_time,

        ROUND(
            EXTRACT(EPOCH FROM (end_time - start_time)) / 3600.0,
            2
        ) AS failure_duration_hours

    FROM failure_events

    ORDER BY event_id
    """
).fetchdf()

print("\nFAILURE EVENTS")
print(windows.to_string(index=False))

# --------------------------------------------------
# Check sensor behaviour before failures
# --------------------------------------------------

pre_failure = con.execute(
    """
    SELECT
        f.event_id,
        f.failure_type,

        ROUND(AVG(m.oil_temperature_mean), 2)
            AS avg_oil_temperature,

        ROUND(MAX(m.oil_temperature_max), 2)
            AS max_oil_temperature,

        ROUND(AVG(m.motor_current_mean), 2)
            AS avg_motor_current,

        ROUND(AVG(m.tp3_mean), 2)
            AS avg_tp3_pressure,

        ROUND(AVG(m.reservoirs_mean), 2)
            AS avg_reservoir_pressure,

        ROUND(AVG(m.flowmeter_mean), 2)
            AS avg_flowmeter

    FROM failure_events f

    JOIN sensor_minute m
        ON m.minute_timestamp >= f.start_time - INTERVAL '24 hours'
       AND m.minute_timestamp < f.start_time

    GROUP BY
        f.event_id,
        f.failure_type

    ORDER BY f.event_id
    """
).fetchdf()

print("\n24 HOURS BEFORE EACH FAILURE")
print(pre_failure.to_string(index=False))

con.close()

elapsed = time.time() - start

print("\n" + "=" * 70)
print("FEATURE LAYER COMPLETE")
print(f"Output: {OUTPUT_PATH}")
print(f"Runtime: {elapsed:.1f} seconds")
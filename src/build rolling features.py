from pathlib import Path
import duckdb
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rolling sensor features.parquet"
)

print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Building rolling time-series features")
print("=" * 70)

start = time.time()

con = duckdb.connect(str(DB_PATH))

con.execute("SET preserve_insertion_order = false")

# --------------------------------------------------
# Rolling features
# All windows look BACKWARD only.
# No future information is used.
# --------------------------------------------------

print("\nCreating rolling sensor features...")

con.execute(
    """
    CREATE OR REPLACE TABLE sensor_rolling_features AS

    WITH rolling AS (

        SELECT
            *,

            /* --------------------------------------
               Window coverage
            -------------------------------------- */

            COUNT(*) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '15 minutes' PRECEDING
                AND CURRENT ROW
            ) AS observations_15m,

            COUNT(*) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS observations_1h,

            COUNT(*) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS observations_6h,

            /* --------------------------------------
               Oil temperature
            -------------------------------------- */

            AVG(oil_temperature_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '15 minutes' PRECEDING
                AND CURRENT ROW
            ) AS oil_temp_avg_15m,

            AVG(oil_temperature_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS oil_temp_avg_1h,

            AVG(oil_temperature_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS oil_temp_avg_6h,

            STDDEV_POP(oil_temperature_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS oil_temp_std_1h,

            STDDEV_POP(oil_temperature_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS oil_temp_std_6h,

            MAX(oil_temperature_max) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS oil_temp_max_1h,

            /* --------------------------------------
               Motor current
            -------------------------------------- */

            AVG(motor_current_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '15 minutes' PRECEDING
                AND CURRENT ROW
            ) AS motor_current_avg_15m,

            AVG(motor_current_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS motor_current_avg_1h,

            AVG(motor_current_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS motor_current_avg_6h,

            STDDEV_POP(motor_current_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS motor_current_std_1h,

            STDDEV_POP(motor_current_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS motor_current_std_6h,

            /* --------------------------------------
               TP3 pressure
            -------------------------------------- */

            AVG(tp3_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '15 minutes' PRECEDING
                AND CURRENT ROW
            ) AS tp3_avg_15m,

            AVG(tp3_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS tp3_avg_1h,

            AVG(tp3_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS tp3_avg_6h,

            STDDEV_POP(tp3_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS tp3_std_1h,

            STDDEV_POP(tp3_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS tp3_std_6h,

            /* --------------------------------------
               Reservoir pressure
            -------------------------------------- */

            AVG(reservoirs_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS reservoir_avg_1h,

            AVG(reservoirs_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS reservoir_avg_6h,

            STDDEV_POP(reservoirs_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS reservoir_std_1h,

            STDDEV_POP(reservoirs_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS reservoir_std_6h,

            /* --------------------------------------
               Air flow
            -------------------------------------- */

            AVG(flowmeter_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS flow_avg_1h,

            AVG(flowmeter_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS flow_avg_6h,

            STDDEV_POP(flowmeter_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS flow_std_1h,

            STDDEV_POP(flowmeter_mean) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS flow_std_6h,

            /* --------------------------------------
               Compressor operating behaviour
            -------------------------------------- */

            AVG(comp_active_rate) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND CURRENT ROW
            ) AS compressor_activity_1h,

            AVG(comp_active_rate) OVER (
                ORDER BY minute_timestamp
                RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                AND CURRENT ROW
            ) AS compressor_activity_6h

        FROM sensor_failure_features
    ),

    engineered AS (

        SELECT
            *,

            /* --------------------------------------
               Temperature trends
            -------------------------------------- */

            oil_temp_avg_15m - oil_temp_avg_1h
                AS oil_temp_short_trend,

            oil_temp_avg_1h - oil_temp_avg_6h
                AS oil_temp_long_trend,

            oil_temperature_mean - oil_temp_avg_6h
                AS oil_temp_deviation_6h,

            (
                oil_temperature_mean - oil_temp_avg_6h
            )
            / NULLIF(oil_temp_std_6h, 0)
                AS oil_temp_zscore_6h,

            /* --------------------------------------
               Motor-current instability
            -------------------------------------- */

            motor_current_avg_15m - motor_current_avg_1h
                AS motor_current_short_trend,

            motor_current_avg_1h - motor_current_avg_6h
                AS motor_current_long_trend,

            (
                motor_current_mean - motor_current_avg_6h
            )
            / NULLIF(motor_current_std_6h, 0)
                AS motor_current_zscore_6h,

            /* --------------------------------------
               Pressure behaviour
            -------------------------------------- */

            tp3_avg_15m - tp3_avg_1h
                AS tp3_short_trend,

            tp3_avg_1h - tp3_avg_6h
                AS tp3_long_trend,

            (
                tp3_mean - tp3_avg_6h
            )
            / NULLIF(tp3_std_6h, 0)
                AS tp3_zscore_6h,

            reservoirs_mean - reservoir_avg_6h
                AS reservoir_deviation_6h,

            (
                reservoirs_mean - reservoir_avg_6h
            )
            / NULLIF(reservoir_std_6h, 0)
                AS reservoir_zscore_6h,

            /* Difference between panel and reservoir pressure */
            ABS(tp3_mean - reservoirs_mean)
                AS pressure_balance_gap,

            /* --------------------------------------
               Air-flow behaviour
            -------------------------------------- */

            flowmeter_mean - flow_avg_6h
                AS flow_deviation_6h,

            (
                flowmeter_mean - flow_avg_6h
            )
            / NULLIF(flow_std_6h, 0)
                AS flow_zscore_6h,

            /* --------------------------------------
               Compressor operating behaviour
            -------------------------------------- */

            compressor_activity_1h
                - compressor_activity_6h
                AS compressor_activity_change,

            /* --------------------------------------
               Within-minute instability
            -------------------------------------- */

            oil_temperature_max
                - oil_temperature_min
                AS oil_temp_range,

            tp3_max - tp3_min
                AS tp3_range,

            reservoirs_max - reservoirs_min
                AS reservoir_range,

            /* --------------------------------------
               Coverage quality
            -------------------------------------- */

            observations_1h / 61.0
                AS coverage_1h,

            observations_6h / 361.0
                AS coverage_6h

        FROM rolling
    )

    SELECT *
    FROM engineered
    ORDER BY minute_timestamp
    """
)

# --------------------------------------------------
# Export
# --------------------------------------------------

print("Exporting rolling feature dataset...")

con.execute(
    f"""
    COPY sensor_rolling_features
    TO '{OUTPUT_PATH.as_posix()}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """
)

# --------------------------------------------------
# Validation
# --------------------------------------------------

summary = con.execute(
    """
    SELECT
        COUNT(*) AS rows,
        COUNT(*) FILTER (
            WHERE eligible_for_model = 1
        ) AS eligible_rows,

        ROUND(AVG(coverage_1h), 3)
            AS avg_1h_coverage,

        ROUND(AVG(coverage_6h), 3)
            AS avg_6h_coverage,

        COUNT(*) FILTER (
            WHERE coverage_6h >= 0.80
        ) AS high_quality_6h_rows

    FROM sensor_rolling_features
    """
).fetchone()

print("\nROLLING FEATURE DATASET")
print(f"Rows:                       {summary[0]:,}")
print(f"Eligible model rows:        {summary[1]:,}")
print(f"Average 1h coverage:        {summary[2]}")
print(f"Average 6h coverage:        {summary[3]}")
print(f"Rows with >=80% 6h data:    {summary[4]:,}")

# --------------------------------------------------
# Compare normal vs pre-failure behaviour
# --------------------------------------------------

comparison = con.execute(
    """
    SELECT

        CASE
            WHEN failure_next_24h = 1
                THEN 'Within 24h of failure'
            ELSE 'Normal'
        END AS period,

        COUNT(*) AS rows,

        ROUND(AVG(oil_temp_long_trend), 3)
            AS oil_temp_trend,

        ROUND(AVG(oil_temp_std_1h), 3)
            AS oil_temp_instability,

        ROUND(AVG(motor_current_std_1h), 3)
            AS current_instability,

        ROUND(AVG(tp3_std_1h), 4)
            AS pressure_instability,

        ROUND(AVG(reservoir_std_1h), 4)
            AS reservoir_instability,

        ROUND(AVG(flow_std_1h), 3)
            AS flow_instability,

        ROUND(AVG(pressure_balance_gap), 4)
            AS pressure_balance_gap

    FROM sensor_rolling_features

    WHERE eligible_for_model = 1
      AND coverage_1h >= 0.50

    GROUP BY 1

    ORDER BY 1
    """
).fetchdf()

print("\nNORMAL VS 24-HOUR PRE-FAILURE BEHAVIOUR")
print(comparison.to_string(index=False))

# --------------------------------------------------
# Extreme feature counts
# --------------------------------------------------

anomalies = con.execute(
    """
    SELECT

        COUNT(*) FILTER (
            WHERE ABS(oil_temp_zscore_6h) >= 2
        ) AS oil_temp_anomalies,

        COUNT(*) FILTER (
            WHERE ABS(motor_current_zscore_6h) >= 2
        ) AS motor_current_anomalies,

        COUNT(*) FILTER (
            WHERE ABS(tp3_zscore_6h) >= 2
        ) AS tp3_anomalies,

        COUNT(*) FILTER (
            WHERE ABS(reservoir_zscore_6h) >= 2
        ) AS reservoir_anomalies,

        COUNT(*) FILTER (
            WHERE ABS(flow_zscore_6h) >= 2
        ) AS flow_anomalies

    FROM sensor_rolling_features

    WHERE eligible_for_model = 1
      AND coverage_6h >= 0.50
    """
).fetchone()

print("\nROLLING Z-SCORE ANOMALIES")
print(f"Oil temperature:    {anomalies[0]:,}")
print(f"Motor current:      {anomalies[1]:,}")
print(f"TP3 pressure:       {anomalies[2]:,}")
print(f"Reservoir pressure: {anomalies[3]:,}")
print(f"Air flow:           {anomalies[4]:,}")

con.close()

elapsed = time.time() - start

print("\n" + "=" * 70)
print("ROLLING FEATURES COMPLETE")
print(f"Output: {OUTPUT_PATH}")
print(f"Runtime: {elapsed:.1f} seconds")
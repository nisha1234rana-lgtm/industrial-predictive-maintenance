from pathlib import Path
import duckdb
import time

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "metropt2" / "MetroPT2.csv"
DB_PATH = PROJECT_ROOT / "industrial maintenance.duckdb"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"MetroPT2 dataset not found: {CSV_PATH}")

# Remove the empty placeholder file created during project setup.
if DB_PATH.exists() and DB_PATH.stat().st_size == 0:
    DB_PATH.unlink()

print("\nINDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE")
print("Building DuckDB warehouse")
print("=" * 70)

start = time.time()

con = duckdb.connect(str(DB_PATH))

con.execute("SET preserve_insertion_order = false")

# --------------------------------------------------
# Raw sensor readings
# --------------------------------------------------

print("\nLoading MetroPT2 sensor readings...")

con.execute(
    f"""
    CREATE OR REPLACE TABLE sensor_readings_raw AS

    SELECT
        timestamp,

        TP2             AS tp2,
        TP3             AS tp3,
        H1              AS h1,
        DV_pressure     AS dv_pressure,
        Reservoirs      AS reservoirs,
        Oil_temperature AS oil_temperature,
        Flowmeter       AS flowmeter,
        Motor_current   AS motor_current,

        COMP            AS comp,
        DV_eletric      AS dv_electric,
        Towers          AS towers,
        MPG             AS mpg,
        LPS             AS lps,
        Pressure_switch AS pressure_switch,
        Oil_level       AS oil_level,
        Caudal_impulses AS caudal_impulses,

        gpsLat          AS gps_lat,
        gpsLong         AS gps_long,
        gpsSpeed        AS gps_speed,
        gpsQuality      AS gps_quality

    FROM read_csv_auto(
        '{CSV_PATH.as_posix()}',
        sample_size = 100000
    )
    """
)

# --------------------------------------------------
# Failure events
# --------------------------------------------------

print("Creating failure event table...")

con.execute("DROP TABLE IF EXISTS failure_events")

con.execute(
    """
    CREATE TABLE failure_events (
        event_id INTEGER,
        failure_type VARCHAR,
        component VARCHAR,
        start_time TIMESTAMP,
        end_time TIMESTAMP
    )
    """
)

failures = [
    (
        1,
        "Air Leak",
        "Air System",
        "2022-06-04 10:19:24.300",
        "2022-06-04 14:22:39.188",
    ),
    (
        2,
        "Oil Leak",
        "Compressor",
        "2022-07-11 10:10:18.948",
        "2022-07-14 10:22:08.046",
    ),
]

con.executemany(
    """
    INSERT INTO failure_events
    VALUES (?, ?, ?, ?, ?)
    """,
    failures,
)

# --------------------------------------------------
# Dataset validation
# --------------------------------------------------

print("\nValidating warehouse...")

summary = con.execute(
    """
    SELECT
        COUNT(*) AS rows,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp,
        COUNT(DISTINCT CAST(timestamp AS DATE)) AS calendar_days,
        COUNT(*) - COUNT(DISTINCT timestamp) AS duplicate_timestamps
    FROM sensor_readings_raw
    """
).fetchone()

row_count = summary[0]
first_timestamp = summary[1]
last_timestamp = summary[2]
calendar_days = summary[3]
duplicate_timestamps = summary[4]

print(f"\nSensor rows:          {row_count:,}")
print(f"First timestamp:      {first_timestamp}")
print(f"Last timestamp:       {last_timestamp}")
print(f"Calendar days:        {calendar_days:,}")
print(f"Duplicate timestamps: {duplicate_timestamps:,}")

# --------------------------------------------------
# Failure validation
# --------------------------------------------------

failure_summary = con.execute(
    """
    SELECT
        f.event_id,
        f.failure_type,
        f.start_time,
        f.end_time,
        COUNT(s.timestamp) AS sensor_rows_during_failure
    FROM failure_events f

    LEFT JOIN sensor_readings_raw s
        ON s.timestamp BETWEEN f.start_time AND f.end_time

    GROUP BY
        f.event_id,
        f.failure_type,
        f.start_time,
        f.end_time

    ORDER BY f.event_id
    """
).fetchdf()

print("\nFAILURE EVENTS")
print(failure_summary.to_string(index=False))

# --------------------------------------------------
# Sensor ranges
# --------------------------------------------------

sensor_ranges = con.execute(
    """
    SELECT
        MIN(tp2) AS tp2_min,
        MAX(tp2) AS tp2_max,

        MIN(tp3) AS tp3_min,
        MAX(tp3) AS tp3_max,

        MIN(reservoirs) AS reservoir_min,
        MAX(reservoirs) AS reservoir_max,

        MIN(oil_temperature) AS oil_temp_min,
        MAX(oil_temperature) AS oil_temp_max,

        MIN(flowmeter) AS flow_min,
        MAX(flowmeter) AS flow_max,

        MIN(motor_current) AS current_min,
        MAX(motor_current) AS current_max

    FROM sensor_readings_raw
    """
).fetchdf()

print("\nCORE SENSOR RANGES")
print(sensor_ranges.to_string(index=False))

con.close()

elapsed = time.time() - start

print("\n" + "=" * 70)
print("WAREHOUSE BUILD COMPLETE")
print(f"Database: {DB_PATH}")
print(f"Runtime: {elapsed:.1f} seconds")
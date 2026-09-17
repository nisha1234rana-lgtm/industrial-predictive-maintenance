from pathlib import Path
import duckdb

DATA_PATH = Path("data/raw/metropt2/MetroPT2.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

con = duckdb.connect()

print("\nMETROPT2 DATA PROFILE")
print("=" * 70)

# Get schema without loading the whole dataset into memory
schema = con.execute(
    f"""
    DESCRIBE
    SELECT *
    FROM read_csv_auto(
        '{DATA_PATH.as_posix()}',
        sample_size=100000
    )
    """
).fetchdf()

print("\nCOLUMNS")
print(schema.to_string(index=False))

# Preview
preview = con.execute(
    f"""
    SELECT *
    FROM read_csv_auto(
        '{DATA_PATH.as_posix()}',
        sample_size=100000
    )
    LIMIT 5
    """
).fetchdf()

print("\nFIRST 5 ROWS")
print(preview.to_string(index=False))

# Total row count
row_count = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_csv_auto(
        '{DATA_PATH.as_posix()}',
        sample_size=100000
    )
    """
).fetchone()[0]

print(f"\nTOTAL ROWS: {row_count:,}")

# Basic timestamp range if timestamp column exists
columns = schema["column_name"].tolist()

timestamp_candidates = [
    col for col in columns
    if "time" in col.lower() or "date" in col.lower()
]

print("\nPOSSIBLE TIMESTAMP COLUMNS")
print(timestamp_candidates)

# Null counts for each column
null_expressions = ",\n".join(
    [
        f'SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "{col}"'
        for col in columns
    ]
)

nulls = con.execute(
    f"""
    SELECT
        {null_expressions}
    FROM read_csv_auto(
        '{DATA_PATH.as_posix()}',
        sample_size=100000
    )
    """
).fetchdf()

print("\nMISSING VALUES")
print(nulls.T.rename(columns={0: "missing_count"}).to_string())

con.close()
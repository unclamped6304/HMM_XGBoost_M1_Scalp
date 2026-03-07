import csv
import psycopg2
import sys
import os

def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

_env = _load_env()

DB_CONFIG = {
    "host": _env.get("DB_HOST", "localhost"),
    "dbname": _env.get("DB_NAME", "postgres"),
    "user": _env.get("DB_USER", "postgres"),
    "password": _env.get("DB_PASSWORD", ""),
}

SCHEMA = "historicalData"
BATCH_SIZE = 10_000

def resolve_table(filepath):
    name = os.path.basename(filepath)
    # Extract pair and timeframe from filename, e.g. GBPUSD_..._M1.csv
    parts = name.replace(".csv", "").split("_")
    pair = parts[0].lower()          # e.g. gbpusd
    timeframe = parts[-1].lower()    # e.g. m1, h4, d1
    return f"{pair}_{timeframe}"

def import_file(filepath):
    table = resolve_table(filepath)
    full_table = f'"{SCHEMA}".{table}'

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    total = 0
    batch = []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip original header

        for row in reader:
            date, time_, open_, high, low, close, tick_vol = row
            # Normalise date: 2016.03.07 -> 2016-03-07
            date = date.replace(".", "-")
            batch.append((date, time_, open_, high, low, close, tick_vol))

            if len(batch) >= BATCH_SIZE:
                cur.executemany(
                    f"INSERT INTO {full_table} (date, time, open, high, low, close, tick_volume) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    batch,
                )
                conn.commit()
                total += len(batch)
                print(f"  Inserted {total} rows...")
                batch = []

        if batch:
            cur.executemany(
                f"INSERT INTO {full_table} (date, time, open, high, low, close, tick_volume) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                batch,
            )
            conn.commit()
            total += len(batch)

    cur.close()
    conn.close()
    print(f"Done. {total} rows imported into {full_table}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_csv.py <path_to_csv>")
        sys.exit(1)
    import_file(sys.argv[1])

from pathlib import Path
import csv
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    r"D:\Part4_Data_ESP32\dataset\processed\iot23_balanced.csv"
)

OUTPUT_DIR = Path(
    r"D:\Part4_Data_ESP32\dataset\processed"
)

TRAIN_FILE = OUTPUT_DIR / "iot23_train.csv"
VAL_FILE = OUTPUT_DIR / "iot23_val.csv"
TEST_FILE = OUTPUT_DIR / "iot23_test.csv"

SCALER_FILE = OUTPUT_DIR / "standard_scaler.joblib"

CHUNK_SIZE = 100_000
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FEATURES = [
    "id.orig_p",
    "id.resp_p",
    "proto",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
]

OUTPUT_COLUMNS = FEATURES + ["class"]


# ============================================================
# CHECK CONFIG
# ============================================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Train/Val/Test ratio must equal 1.0")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# PASS 1
# Fit StandardScaler using TRAIN samples only
# ============================================================

print("=" * 60)
print("PASS 1: FIT STANDARD SCALER")
print("=" * 60)

scaler = StandardScaler()

total_rows = 0
train_rows = 0
val_rows = 0
test_rows = 0

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:

    reader = csv.DictReader(f)

    for chunk_start in range(0, 3_000_000, CHUNK_SIZE):

        rows = []

        for _ in range(CHUNK_SIZE):

            try:
                row = next(reader)
            except StopIteration:
                break

            rows.append(row)

        if not rows:
            break

        n = len(rows)

        X = np.array(
            [
                [float(row[feature]) for feature in FEATURES]
                for row in rows
            ],
            dtype=np.float64,
        )

        # Random split
        r = rng.random(n)

        train_mask = r < TRAIN_RATIO
        val_mask = (
            (r >= TRAIN_RATIO)
            & (r < TRAIN_RATIO + VAL_RATIO)
        )
        test_mask = r >= TRAIN_RATIO + VAL_RATIO

        if np.any(train_mask):
            scaler.partial_fit(X[train_mask])
            train_rows += int(np.sum(train_mask))

        val_rows += int(np.sum(val_mask))
        test_rows += int(np.sum(test_mask))

        total_rows += n

        print(
            f"\rProcessed: {total_rows:,} / 3,000,000",
            end=""
        )

print()

print()
print(f"Total rows : {total_rows:,}")
print(f"Train      : {train_rows:,}")
print(f"Validation : {val_rows:,}")
print(f"Test       : {test_rows:,}")

print()
print("Scaler mean:")
for feature, value in zip(FEATURES, scaler.mean_):
    print(f"    {feature:15s}: {value:.6g}")

print()
print("Scaler std:")
for feature, value in zip(FEATURES, scaler.scale_):
    print(f"    {feature:15s}: {value:.6g}")


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(scaler, SCALER_FILE)

print()
print(f"Scaler saved: {SCALER_FILE}")


# ============================================================
# PASS 2
# Normalize and write Train / Val / Test
# ============================================================

print()
print("=" * 60)
print("PASS 2: NORMALIZE DATASET")
print("=" * 60)

# Reinitialize RNG so the exact same split is generated
rng = np.random.default_rng(SEED)

# Remove old files if they exist
for file in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
    if file.exists():
        file.unlink()


train_out = open(
    TRAIN_FILE,
    "w",
    encoding="utf-8",
    newline="",
    buffering=1024 * 1024,
)

val_out = open(
    VAL_FILE,
    "w",
    encoding="utf-8",
    newline="",
    buffering=1024 * 1024,
)

test_out = open(
    TEST_FILE,
    "w",
    encoding="utf-8",
    newline="",
    buffering=1024 * 1024,
)

train_writer = csv.writer(train_out)
val_writer = csv.writer(val_out)
test_writer = csv.writer(test_out)

train_writer.writerow(OUTPUT_COLUMNS)
val_writer.writerow(OUTPUT_COLUMNS)
test_writer.writerow(OUTPUT_COLUMNS)

total_rows = 0
written_train = 0
written_val = 0
written_test = 0

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:

    reader = csv.DictReader(f)

    while True:

        rows = []

        for _ in range(CHUNK_SIZE):

            try:
                row = next(reader)
            except StopIteration:
                break

            rows.append(row)

        if not rows:
            break

        n = len(rows)

        X = np.array(
            [
                [float(row[feature]) for feature in FEATURES]
                for row in rows
            ],
            dtype=np.float64,
        )

        y = np.array(
            [int(row["class"]) for row in rows],
            dtype=np.int8,
        )

        # Same split as PASS 1
        r = rng.random(n)

        train_mask = r < TRAIN_RATIO

        val_mask = (
            (r >= TRAIN_RATIO)
            & (r < TRAIN_RATIO + VAL_RATIO)
        )

        test_mask = r >= TRAIN_RATIO + VAL_RATIO

        # Normalize
        X_scaled = scaler.transform(X)

        # Save as float32 to reduce file size
        X_scaled = X_scaled.astype(np.float32)

        # -------------------------
        # TRAIN
        # -------------------------

        train_indices = np.where(train_mask)[0]

        for i in train_indices:
            train_writer.writerow(
                list(X_scaled[i]) + [int(y[i])]
            )

        written_train += len(train_indices)

        # -------------------------
        # VALIDATION
        # -------------------------

        val_indices = np.where(val_mask)[0]

        for i in val_indices:
            val_writer.writerow(
                list(X_scaled[i]) + [int(y[i])]
            )

        written_val += len(val_indices)

        # -------------------------
        # TEST
        # -------------------------

        test_indices = np.where(test_mask)[0]

        for i in test_indices:
            test_writer.writerow(
                list(X_scaled[i]) + [int(y[i])]
            )

        written_test += len(test_indices)

        total_rows += n

        print(
            f"\rProcessed: {total_rows:,} / 3,000,000",
            end=""
        )

train_out.close()
val_out.close()
test_out.close()

print()
print()
print("=" * 60)
print("NORMALIZATION FINISHED")
print("=" * 60)

print(f"Train      : {written_train:,}")
print(f"Validation : {written_val:,}")
print(f"Test       : {written_test:,}")
print(f"Total      : {written_train + written_val + written_test:,}")

print()
print("Output files:")
print(f"  {TRAIN_FILE}")
print(f"  {VAL_FILE}")
print(f"  {TEST_FILE}")
print(f"  {SCALER_FILE}")

print()
print("=" * 60)

if written_train + written_val + written_test == 3_000_000:
    print("NORMALIZATION: PASS")
else:
    print("NORMALIZATION: FAIL")

print("=" * 60)
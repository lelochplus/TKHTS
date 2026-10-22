from pathlib import Path
import csv
import random
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(r"D:\Part4_Data_ESP32")

INPUT_FILE = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "iot23_balanced_behavior.csv"
)

OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

TRAIN_FILE = OUTPUT_DIR / "iot23_behavior_train.csv"
VAL_FILE = OUTPUT_DIR / "iot23_behavior_val.csv"
TEST_FILE = OUTPUT_DIR / "iot23_behavior_test.csv"

SCALER_FILE = (
    OUTPUT_DIR
    / "standard_scaler_behavior.joblib"
)

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CHUNK_SIZE = 100_000


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

    "conn_count_10s",
    "unique_dst_ports_10s",
    "unique_dst_ips_10s",
    "failed_conn_count_10s",
    "failed_ratio_10s",
    "s0_ratio_10s",
]

TARGET = "class"


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("IoT-23 BEHAVIOR DATASET")
print("SPLIT FIRST + SCALER FIT ON TRAIN ONLY")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}"
    )

print()
print(f"Input : {INPUT_FILE}")
print(
    f"Size  : "
    f"{INPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB"
)


# ============================================================
# PASS 1
# COUNT TOTAL ROWS
# ============================================================

print()
print("=" * 70)
print("PASS 1: COUNT DATASET")
print("=" * 70)

total_rows = 0

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline="",
    buffering=1024 * 1024
) as f:

    reader = csv.DictReader(f)

    if reader.fieldnames is None:
        raise RuntimeError("CSV header not found.")

    missing = [
        x for x in FEATURES + [TARGET]
        if x not in reader.fieldnames
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    for _ in reader:
        total_rows += 1

print(f"Total rows : {total_rows:,}")


# ============================================================
# DETERMINE SPLIT COUNTS
# ============================================================

train_target = int(total_rows * TRAIN_RATIO)
val_target = int(total_rows * VAL_RATIO)
test_target = total_rows - train_target - val_target

print()
print("Target split:")
print(f"    Train      : {train_target:,}")
print(f"    Validation : {val_target:,}")
print(f"    Test       : {test_target:,}")


# ============================================================
# PASS 2
# CREATE RANDOM SPLIT
# ============================================================

print()
print("=" * 70)
print("PASS 2: CREATE RANDOM SPLIT")
print("=" * 70)

# Store randomized row assignment only.
# The actual feature data remains in the original CSV.

indices = list(range(total_rows))

random.Random(RANDOM_SEED).shuffle(indices)

train_indices = set(indices[:train_target])
val_indices = set(
    indices[train_target:train_target + val_target]
)

# Remaining rows belong to test.
# We do not need to store test_indices.


print("Random shuffle complete.")


# ============================================================
# PASS 3
# FIT SCALER ON TRAIN ONLY
# ============================================================

print()
print("=" * 70)
print("PASS 3: FIT STANDARD SCALER ON TRAIN ONLY")
print("=" * 70)

scaler = StandardScaler()

train_seen = 0
processed = 0

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline="",
    buffering=1024 * 1024
) as f:

    reader = csv.DictReader(f)

    chunk = []

    for index, row in enumerate(reader):

        if index not in train_indices:
            continue

        values = [
            float(row[feature])
            for feature in FEATURES
        ]

        chunk.append(values)

        if len(chunk) >= CHUNK_SIZE:

            scaler.partial_fit(
                np.asarray(
                    chunk,
                    dtype=np.float64
                )
            )

            train_seen += len(chunk)
            chunk = []

            print(
                f"\rTrain scaler: "
                f"{train_seen:,} / "
                f"{train_target:,}",
                end="",
                flush=True
            )

    if chunk:

        scaler.partial_fit(
            np.asarray(
                chunk,
                dtype=np.float64
            )
        )

        train_seen += len(chunk)

print()

print()
print(f"Train rows used for scaler: {train_seen:,}")

if train_seen != train_target:
    raise RuntimeError(
        "Scaler training row count mismatch."
    )


print()
print("Scaler mean:")

for feature, value in zip(
    FEATURES,
    scaler.mean_
):
    print(
        f"    {feature:24s}: "
        f"{value:.6g}"
    )


print()
print("Scaler std:")

for feature, value in zip(
    FEATURES,
    scaler.scale_
):
    print(
        f"    {feature:24s}: "
        f"{value:.6g}"
    )


joblib.dump(
    scaler,
    SCALER_FILE
)

print()
print("Scaler saved:")
print(f"    {SCALER_FILE}")


# ============================================================
# PASS 4
# NORMALIZE + WRITE TRAIN / VAL / TEST
# ============================================================

print()
print("=" * 70)
print("PASS 4: NORMALIZE + WRITE DATASETS")
print("=" * 70)


for file in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:

    if file.exists():
        print(f"Removing old file: {file}")
        file.unlink()


train_file = open(
    TRAIN_FILE,
    "w",
    newline="",
    encoding="utf-8",
    buffering=1024 * 1024
)

val_file = open(
    VAL_FILE,
    "w",
    newline="",
    encoding="utf-8",
    buffering=1024 * 1024
)

test_file = open(
    TEST_FILE,
    "w",
    newline="",
    encoding="utf-8",
    buffering=1024 * 1024
)


train_writer = csv.writer(train_file)
val_writer = csv.writer(val_file)
test_writer = csv.writer(test_file)


header = FEATURES + [TARGET]

train_writer.writerow(header)
val_writer.writerow(header)
test_writer.writerow(header)


train_count = 0
val_count = 0
test_count = 0

class_train = {0: 0, 1: 0, 2: 0}
class_val = {0: 0, 1: 0, 2: 0}
class_test = {0: 0, 1: 0, 2: 0}


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline="",
    buffering=1024 * 1024
) as f:

    reader = csv.DictReader(f)

    for index, row in enumerate(reader):

        values = np.asarray(
            [
                float(row[feature])
                for feature in FEATURES
            ],
            dtype=np.float64
        )

        normalized = scaler.transform(
            values.reshape(1, -1)
        )[0]

        label = int(row[TARGET])

        output_row = (
            normalized.tolist()
            + [label]
        )

        if index in train_indices:

            train_writer.writerow(output_row)

            train_count += 1
            class_train[label] += 1

        elif index in val_indices:

            val_writer.writerow(output_row)

            val_count += 1
            class_val[label] += 1

        else:

            test_writer.writerow(output_row)

            test_count += 1
            class_test[label] += 1

        processed += 1

        if processed % 100_000 == 0:

            print(
                f"\rProcessed: "
                f"{processed:,} / "
                f"{total_rows:,}",
                end="",
                flush=True
            )


train_file.close()
val_file.close()
test_file.close()


# ============================================================
# FINAL CHECK
# ============================================================

print()
print()
print("=" * 70)
print("NORMALIZATION FINISHED")
print("=" * 70)

print()
print(f"Train      : {train_count:,}")
print(f"Validation : {val_count:,}")
print(f"Test       : {test_count:,}")

print(
    f"Total      : "
    f"{train_count + val_count + test_count:,}"
)


print()
print("Class distribution:")

print()
print("Train:")
print(f"    Normal    : {class_train[0]:,}")
print(f"    DoS       : {class_train[1]:,}")
print(f"    Port Scan : {class_train[2]:,}")

print()
print("Validation:")
print(f"    Normal    : {class_val[0]:,}")
print(f"    DoS       : {class_val[1]:,}")
print(f"    Port Scan : {class_val[2]:,}")

print()
print("Test:")
print(f"    Normal    : {class_test[0]:,}")
print(f"    DoS       : {class_test[1]:,}")
print(f"    Port Scan : {class_test[2]:,}")


if (
    train_count
    + val_count
    + test_count
    != total_rows
):
    raise RuntimeError(
        "Final row count mismatch."
    )


print()
print("Output files:")
print(f"    {TRAIN_FILE}")
print(f"    {VAL_FILE}")
print(f"    {TEST_FILE}")
print(f"    {SCALER_FILE}")

print()
print(f"FEATURE COUNT : {len(FEATURES)}")
print(f"INPUT DIMENSION: {len(FEATURES)}")

print()
print("DATA LEAKAGE CHECK:")
print("    Scaler fitted on TRAIN ONLY")
print("    Validation/Test only transformed")

print()
print("NORMALIZATION: PASS")
print("=" * 70)
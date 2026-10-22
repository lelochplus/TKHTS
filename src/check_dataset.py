from pathlib import Path
import csv

FILE = Path(r"D:\Part4_Data_ESP32\dataset\processed\iot23_balanced.csv")

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

EXPECTED_COLUMNS = FEATURES + ["class"]

class_names = {
    0: "Normal",
    1: "DoS",
    2: "Port Scan",
}

count = 0
class_count = {0: 0, 1: 0, 2: 0}
empty_count = 0
invalid_count = 0

min_values = {f: float("inf") for f in FEATURES}
max_values = {f: float("-inf") for f in FEATURES}

print("=" * 60)
print("IoT-23 BALANCED DATASET CHECK")
print("=" * 60)

with open(FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    print("Columns:")
    print(reader.fieldnames)
    print()

    if reader.fieldnames != EXPECTED_COLUMNS:
        print("ERROR: Columns không đúng!")
        print("Expected:")
        print(EXPECTED_COLUMNS)
        raise SystemExit(1)

    for row in reader:
        count += 1

        # -------------------------
        # Check class
        # -------------------------
        try:
            label = int(row["class"])
        except (ValueError, TypeError):
            invalid_count += 1
            continue

        if label not in class_count:
            invalid_count += 1
            continue

        class_count[label] += 1

        # -------------------------
        # Check features
        # -------------------------
        for feature in FEATURES:
            value = row[feature]

            if value == "":
                empty_count += 1
                continue

            try:
                x = float(value)
            except ValueError:
                invalid_count += 1
                continue

            if x < min_values[feature]:
                min_values[feature] = x

            if x > max_values[feature]:
                max_values[feature] = x


print("=" * 60)
print("RESULT")
print("=" * 60)

print(f"Total rows : {count:,}")
print()

print("Class distribution:")
for label in [0, 1, 2]:
    print(f"    {label} = {class_names[label]:10s}: {class_count[label]:,}")

print()
print(f"Empty values   : {empty_count:,}")
print(f"Invalid values : {invalid_count:,}")

print()
print("Feature ranges:")
for feature in FEATURES:
    print(
        f"    {feature:15s}: "
        f"min={min_values[feature]:.6g} "
        f"max={max_values[feature]:.6g}"
    )

print()
print("=" * 60)

if (
    count == 3_000_000
    and class_count[0] == 1_000_000
    and class_count[1] == 1_000_000
    and class_count[2] == 1_000_000
    and empty_count == 0
    and invalid_count == 0
):
    print("DATASET CHECK: PASS")
else:
    print("DATASET CHECK: FAIL")

print("=" * 60)
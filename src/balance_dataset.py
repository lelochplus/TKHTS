from pathlib import Path
import csv
import random

# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path(
    r"C:\IoT23\raw\opt\Malware-Project\BigDataset\IoTScenarios"
)

OUTPUT = Path(
    r"D:\Part4_Data_ESP32\dataset\processed\iot23_balanced.csv"
)

# 1,000,000 mẫu cho mỗi class
SAMPLES_PER_CLASS = 1_000_000

# Random seed để lần chạy khác nhau vẫn có thể tái lập
SEED = 42


# ============================================================
# FEATURE
# ============================================================

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
# MAPPING
# ============================================================

PROTO_MAP = {
    "tcp": 0,
    "udp": 1,
    "icmp": 2,
}

STATE_MAP = {
    "S0": 0,
    "S1": 1,
    "SF": 2,
    "REJ": 3,
    "RSTO": 4,
    "RSTR": 5,
    "SH": 6,
    "OTH": 7,
}


# ============================================================
# CONVERT NUMBER
# ============================================================

def to_number(value):
    """
    Chuyển dữ liệu số của IoT-23 sang float.

    IoT-23 có:
        -
        (empty)
        số thực
        số nguyên
    """

    value = value.strip()

    if value in ("", "-", "(empty)"):
        return 0.0

    return float(value)


# ============================================================
# GET CLASS
# ============================================================

def get_class(record):
    """
    Class:
        0 = Normal
        1 = DoS
        2 = Port Scan
    """

    label = record.get("label", "").strip()
    detailed = record.get("detailed-label", "").strip()

    # Normal
    if label == "Benign":
        return 0

    # DoS
    if detailed == "DDoS":
        return 1

    # Port Scan
    if detailed == "PartOfAHorizontalPortScan":
        return 2

    return None


# ============================================================
# CONVERT RECORD
# ============================================================

def convert_record(record, class_id):

    try:

        proto = record["proto"].strip().lower()
        state = record["conn_state"].strip()

        # Không lấy protocol/state ngoài mapping
        if proto not in PROTO_MAP:
            return None

        if state not in STATE_MAP:
            return None

        row = [
            int(to_number(record["id.orig_p"])),
            int(to_number(record["id.resp_p"])),

            PROTO_MAP[proto],

            to_number(record["duration"]),
            to_number(record["orig_bytes"]),
            to_number(record["resp_bytes"]),

            STATE_MAP[state],

            to_number(record["orig_pkts"]),
            to_number(record["resp_pkts"]),

            to_number(record["orig_ip_bytes"]),
            to_number(record["resp_ip_bytes"]),

            class_id,
        ]

        return row

    except (KeyError, ValueError):

        return None


# ============================================================
# RESERVOIR SAMPLING
# ============================================================

def reservoir_add(reservoir, item, count, rng):

    # Chưa đủ 1 triệu mẫu
    if len(reservoir) < SAMPLES_PER_CLASS:

        reservoir.append(item)

    else:

        # Reservoir sampling
        index = rng.randint(0, count - 1)

        if index < SAMPLES_PER_CLASS:

            reservoir[index] = item


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path, reservoirs, seen, rng):

    records = 0
    selected = 0
    skipped = 0

    header = None

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
        buffering=1024 * 1024,
    ) as f:

        for line in f:

            line = line.rstrip("\r\n")

            if not line:
                continue

            # ------------------------------------------------
            # Header #fields
            # ------------------------------------------------

            if line.startswith("#fields"):

                # IoT-23 dùng whitespace
                parts = line.split()

                header = parts[1:]

                continue

            # ------------------------------------------------
            # Các dòng metadata
            # ------------------------------------------------

            if line.startswith("#"):
                continue

            # Chưa tìm thấy #fields
            if header is None:
                continue

            # ------------------------------------------------
            # Data
            # ------------------------------------------------

            # Quan trọng:
            # IoT-23 thực tế dùng whitespace ở phần label.
            fields = line.split()

            # Số cột phải khớp header
            if len(fields) != len(header):

                skipped += 1
                continue

            records += 1

            record = dict(
                zip(header, fields)
            )

            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            class_id = get_class(record)

            if class_id is None:
                continue

            # ------------------------------------------------
            # Convert feature
            # ------------------------------------------------

            row = convert_record(
                record,
                class_id
            )

            if row is None:

                skipped += 1
                continue

            # ------------------------------------------------
            # Reservoir
            # ------------------------------------------------

            seen[class_id] += 1

            reservoir_add(
                reservoirs[class_id],
                row,
                seen[class_id],
                rng
            )

            selected += 1

    return records, selected, skipped


# ============================================================
# MAIN
# ============================================================

def main():

    rng = random.Random(SEED)

    # Reservoir cho 3 class
    reservoirs = {
        0: [],
        1: [],
        2: [],
    }

    # Tổng số mẫu đã gặp của từng class
    seen = {
        0: 0,
        1: 0,
        2: 0,
    }

    # --------------------------------------------------------
    # Tìm toàn bộ conn.log.labeled
    # --------------------------------------------------------

    files = sorted(
        RAW_DIR.rglob("conn.log.labeled")
    )

    print("=" * 60)
    print("IoT-23 Balanced Dataset Sampling")
    print("=" * 60)

    print(
        f"Raw directory : {RAW_DIR}"
    )

    print(
        f"Files found   : {len(files)}"
    )

    print(
        f"Samples/class : {SAMPLES_PER_CLASS:,}"
    )

    print()

    # --------------------------------------------------------
    # Process 23 scenarios
    # --------------------------------------------------------

    for index, file_path in enumerate(files, 1):

        scenario = file_path.parent.parent.name

        print(
            f"[{index}/{len(files)}] {scenario}"
        )

        records, selected, skipped = process_file(
            file_path,
            reservoirs,
            seen,
            rng
        )

        print(
            f"    Records : {records:,}"
        )

        print(
            f"    Selected: {selected:,}"
        )

        print(
            f"    Skipped : {skipped:,}"
        )

        print(
            f"    Reservoir -> "
            f"Normal={len(reservoirs[0]):,}, "
            f"DoS={len(reservoirs[1]):,}, "
            f"PortScan={len(reservoirs[2]):,}"
        )

        print()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("=" * 60)
    print("Sampling finished")
    print("=" * 60)

    print(
        f"Normal    : {len(reservoirs[0]):,}"
    )

    print(
        f"DoS       : {len(reservoirs[1]):,}"
    )

    print(
        f"Port Scan : {len(reservoirs[2]):,}"
    )

    print()

    # ========================================================
    # CHECK
    # ========================================================

    for class_id in (0, 1, 2):

        if len(reservoirs[class_id]) < SAMPLES_PER_CLASS:

            raise RuntimeError(
                f"Class {class_id} không đủ "
                f"{SAMPLES_PER_CLASS:,} mẫu."
            )

    # ========================================================
    # MERGE
    # ========================================================

    print("Merging dataset...")

    rows = (
        reservoirs[0]
        + reservoirs[1]
        + reservoirs[2]
    )

    # Trộn ngẫu nhiên
    print("Shuffling dataset...")

    rng.shuffle(rows)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    print()
    print(
        f"Writing: {OUTPUT}"
    )

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as f:

        writer = csv.writer(f)

        # Header
        writer.writerow(
            OUTPUT_COLUMNS
        )

        # 3,000,000 rows
        writer.writerows(rows)

    # ========================================================
    # FINAL
    # ========================================================

    file_size_mb = (
        OUTPUT.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Output : {OUTPUT}"
    )

    print(
        f"Rows   : {len(rows):,}"
    )

    print(
        f"Size   : {file_size_mb:.2f} MB"
    )

    print()
    print("Class distribution:")

    print(
        "    Normal    = 1,000,000"
    )

    print(
        "    DoS       = 1,000,000"
    )

    print(
        "    Port Scan = 1,000,000"
    )

    print(
        "    TOTAL     = 3,000,000"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
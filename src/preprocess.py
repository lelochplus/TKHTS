from pathlib import Path
from collections import defaultdict, deque
import csv
import random

# ============================================================
# IoT-23 PREPROCESSING
# 11 ORIGINAL FEATURES + 6 BEHAVIORAL FEATURES
#
# Optimized version:
# - Sliding window 10 seconds
# - Incremental counters
# - No full scan of the window for every record
# - Reservoir sampling: 1,000,000 samples/class
# ============================================================

RAW_DIR = Path(r"C:\IoT23\raw")
OUTPUT_DIR = Path(r"D:\Part4_Data_ESP32\dataset\processed")

OUTPUT_FILE = OUTPUT_DIR / "iot23_balanced_behavior.csv"

SAMPLES_PER_CLASS = 1_000_000
WINDOW_SECONDS = 10.0
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)


# ============================================================
# ORIGINAL FEATURES
# ============================================================

ORIGINAL_FEATURES = [
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


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

BEHAVIOR_FEATURES = [
    "conn_count_10s",
    "unique_dst_ports_10s",
    "unique_dst_ips_10s",
    "failed_conn_count_10s",
    "failed_ratio_10s",
    "s0_ratio_10s",
]


FEATURES = ORIGINAL_FEATURES + BEHAVIOR_FEATURES


# ============================================================
# PROTOCOL
# ============================================================

PROTO_MAP = {
    "tcp": 0,
    "udp": 1,
    "icmp": 2,
}


# ============================================================
# CONNECTION STATE
# ============================================================

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


# States considered failed
FAILED_STATES = {
    "S0",
    "REJ",
    "RSTO",
    "RSTR",
    "SH",
}


# ============================================================
# CONVERT VALUE TO NUMBER
# ============================================================

def to_number(value):

    if value in ["-", "(empty)", ""]:
        return 0.0

    try:
        return float(value)

    except ValueError:
        return 0.0


# ============================================================
# CLASSIFICATION
#
# 0 = Normal
# 1 = DoS
# 2 = Port Scan
# ============================================================

def get_class(record):

    label = record.get("label", "")
    detailed = record.get("detailed-label", "")

    if label == "Benign":
        return 0

    if detailed == "DDoS":
        return 1

    if detailed == "PartOfAHorizontalPortScan":
        return 2

    return None


# ============================================================
# FIND ALL IoT-23 FILES
# ============================================================

files = sorted(
    RAW_DIR.rglob("conn.log.labeled")
)


print("=" * 70)
print("IoT-23 PREPROCESSING - OPTIMIZED")
print("11 ORIGINAL + 6 BEHAVIORAL FEATURES")
print("=" * 70)

print(f"Found files : {len(files)}")

if len(files) != 23:
    print("WARNING: Expected 23 conn.log.labeled files!")


# ============================================================
# RESERVOIR SAMPLING
# ============================================================

normal_samples = []
dos_samples = []
portscan_samples = []


def reservoir_add(reservoir, row, count):

    if len(reservoir) < SAMPLES_PER_CLASS:

        reservoir.append(row)

    else:

        index = random.randrange(count)

        if index < SAMPLES_PER_CLASS:
            reservoir[index] = row


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total = 0
selected = 0
skipped = 0

normal_count = 0
dos_count = 0
portscan_count = 0


# ============================================================
# PROCESS SCENARIOS
# ============================================================

for file_index, file_path in enumerate(files, start=1):

    print()
    print("-" * 70)

    print(
        f"[{file_index}/{len(files)}] "
        f"{file_path.parent.parent.name}"
    )

    field_names = None

    data_count = 0
    selected_count = 0
    skipped_count = 0

    # ========================================================
    # SLIDING WINDOWS
    #
    # source_ip -> deque
    #
    # Each deque entry:
    #
    # (
    #   timestamp,
    #   dst_port,
    #   dst_ip,
    #   failed,
    #   is_s0
    # )
    # ========================================================

    windows = defaultdict(deque)

    # ========================================================
    # INCREMENTAL COUNTERS
    #
    # These avoid scanning the entire deque.
    # ========================================================

    port_counts = defaultdict(lambda: defaultdict(int))
    ip_counts = defaultdict(lambda: defaultdict(int))

    conn_counts = defaultdict(int)
    failed_counts = defaultdict(int)
    s0_counts = defaultdict(int)

    # ========================================================
    # READ FILE
    # ========================================================

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
        buffering=1024 * 1024
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            if line.startswith("#fields"):

                field_names = line.split()[1:]

                continue

            # ------------------------------------------------
            # Other metadata
            # ------------------------------------------------

            if line.startswith("#"):
                continue

            # ------------------------------------------------
            # Check header
            # ------------------------------------------------

            if field_names is None:

                skipped += 1
                skipped_count += 1

                continue

            # ------------------------------------------------
            # Split record
            # ------------------------------------------------

            fields = line.split()

            if len(fields) < len(field_names):

                skipped += 1
                skipped_count += 1

                continue

            record = dict(
                zip(field_names, fields)
            )

            total += 1
            data_count += 1

            # =================================================
            # TIMESTAMP
            # =================================================

            try:

                timestamp = float(
                    record.get("ts", "0")
                )

            except ValueError:

                timestamp = 0.0

            # =================================================
            # SOURCE / DESTINATION
            # =================================================

            src_ip = record.get(
                "id.orig_h",
                "unknown"
            )

            dst_ip = record.get(
                "id.resp_h",
                "unknown"
            )

            dst_port_text = record.get(
                "id.resp_p",
                "0"
            )

            try:

                dst_port = int(
                    float(dst_port_text)
                )

            except ValueError:

                dst_port = 0

            conn_state = record.get(
                "conn_state",
                "-"
            )

            is_failed = (
                conn_state in FAILED_STATES
            )

            is_s0 = (
                conn_state == "S0"
            )

            # =================================================
            # GET SOURCE WINDOW
            # =================================================

            window = windows[src_ip]

            # =================================================
            # REMOVE OLD CONNECTIONS
            #
            # IMPORTANT:
            # Remove BEFORE adding current connection.
            # =================================================

            min_timestamp = (
                timestamp - WINDOW_SECONDS
            )

            while window:

                old = window[0]

                old_timestamp = old[0]

                if old_timestamp >= min_timestamp:
                    break

                window.popleft()

                old_port = old[1]
                old_ip = old[2]
                old_failed = old[3]
                old_s0 = old[4]

                # Remove port count

                port_counts[src_ip][old_port] -= 1

                if port_counts[src_ip][old_port] <= 0:

                    del port_counts[src_ip][old_port]

                # Remove IP count

                ip_counts[src_ip][old_ip] -= 1

                if ip_counts[src_ip][old_ip] <= 0:

                    del ip_counts[src_ip][old_ip]

                # Update counters

                conn_counts[src_ip] -= 1

                if old_failed:
                    failed_counts[src_ip] -= 1

                if old_s0:
                    s0_counts[src_ip] -= 1

            # =================================================
            # ADD CURRENT CONNECTION
            # =================================================

            window.append(
                (
                    timestamp,
                    dst_port,
                    dst_ip,
                    is_failed,
                    is_s0,
                )
            )

            port_counts[src_ip][dst_port] += 1

            ip_counts[src_ip][dst_ip] += 1

            conn_counts[src_ip] += 1

            if is_failed:
                failed_counts[src_ip] += 1

            if is_s0:
                s0_counts[src_ip] += 1

            # =================================================
            # BEHAVIOR FEATURES
            # =================================================

            conn_count_10s = conn_counts[src_ip]

            unique_dst_ports_10s = len(
                port_counts[src_ip]
            )

            unique_dst_ips_10s = len(
                ip_counts[src_ip]
            )

            failed_conn_count_10s = (
                failed_counts[src_ip]
            )

            s0_count_10s = (
                s0_counts[src_ip]
            )

            if conn_count_10s > 0:

                failed_ratio_10s = (
                    failed_conn_count_10s
                    / conn_count_10s
                )

                s0_ratio_10s = (
                    s0_count_10s
                    / conn_count_10s
                )

            else:

                failed_ratio_10s = 0.0
                s0_ratio_10s = 0.0

            # =================================================
            # CLASS
            # =================================================

            label = get_class(record)

            if label is None:
                continue

            # =================================================
            # ORIGINAL FEATURES
            # =================================================

            try:

                row = []

                for feature in ORIGINAL_FEATURES:

                    value = record.get(
                        feature,
                        "0"
                    )

                    if feature == "proto":

                        value = PROTO_MAP.get(
                            value,
                            0
                        )

                    elif feature == "conn_state":

                        value = STATE_MAP.get(
                            value,
                            0
                        )

                    else:

                        value = to_number(value)

                    row.append(value)

                # =================================================
                # BEHAVIOR FEATURES
                # =================================================

                row.extend([
                    float(conn_count_10s),
                    float(unique_dst_ports_10s),
                    float(unique_dst_ips_10s),
                    float(failed_conn_count_10s),
                    float(failed_ratio_10s),
                    float(s0_ratio_10s),
                ])

                # =================================================
                # LABEL
                # =================================================

                row.append(label)

                # =================================================
                # RESERVOIR SAMPLE
                # =================================================

                if label == 0:

                    normal_count += 1

                    reservoir_add(
                        normal_samples,
                        row,
                        normal_count
                    )

                elif label == 1:

                    dos_count += 1

                    reservoir_add(
                        dos_samples,
                        row,
                        dos_count
                    )

                elif label == 2:

                    portscan_count += 1

                    reservoir_add(
                        portscan_samples,
                        row,
                        portscan_count
                    )

                selected += 1
                selected_count += 1

            except Exception:

                skipped += 1
                skipped_count += 1

    # ========================================================
    # SCENARIO RESULT
    # ========================================================

    print(
        f"    Records  : {data_count:,}"
    )

    print(
        f"    Selected : {selected_count:,}"
    )

    print(
        f"    Skipped  : {skipped_count:,}"
    )

    print(
        f"    Samples  : "
        f"N={len(normal_samples):,}, "
        f"D={len(dos_samples):,}, "
        f"PS={len(portscan_samples):,}"
    )


# ============================================================
# SAMPLING RESULT
# ============================================================

print()
print("=" * 70)
print("SAMPLING FINISHED")
print("=" * 70)

print(
    f"Normal reservoir    : "
    f"{len(normal_samples):,}"
)

print(
    f"DoS reservoir       : "
    f"{len(dos_samples):,}"
)

print(
    f"Port Scan reservoir : "
    f"{len(portscan_samples):,}"
)


if len(normal_samples) < SAMPLES_PER_CLASS:

    raise RuntimeError(
        "Not enough Normal samples."
    )


if len(dos_samples) < SAMPLES_PER_CLASS:

    raise RuntimeError(
        "Not enough DoS samples."
    )


if len(portscan_samples) < SAMPLES_PER_CLASS:

    raise RuntimeError(
        "Not enough Port Scan samples."
    )


# ============================================================
# MERGE
# ============================================================

print()
print("Merging dataset...")

all_samples = (
    normal_samples
    + dos_samples
    + portscan_samples
)


# ============================================================
# SHUFFLE
# ============================================================

print("Shuffling dataset...")

random.shuffle(all_samples)


# ============================================================
# WRITE OUTPUT
# ============================================================

print()
print("Writing:")

print(OUTPUT_FILE)

if OUTPUT_FILE.exists():

    print("Removing old output...")

    OUTPUT_FILE.unlink()


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
    buffering=1024 * 1024
) as output:

    writer = csv.writer(output)

    writer.writerow(
        FEATURES + ["class"]
    )

    writer.writerows(
        all_samples
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"Total raw records : "
    f"{total:,}"
)

print(
    f"Selected records  : "
    f"{selected:,}"
)

print(
    f"Skipped records   : "
    f"{skipped:,}"
)

print()
print("Class distribution:")

print(
    f"    Normal    : "
    f"{len(normal_samples):,}"
)

print(
    f"    DoS       : "
    f"{len(dos_samples):,}"
)

print(
    f"    Port Scan : "
    f"{len(portscan_samples):,}"
)

print(
    f"    TOTAL     : "
    f"{len(all_samples):,}"
)

print()
print("Features:")

for index, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"    {index:2d}. {feature}"
    )

print()
print(
    f"Output: {OUTPUT_FILE}"
)

print()
print("SUCCESS.")
print("=" * 70)
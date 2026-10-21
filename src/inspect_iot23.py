import tarfile
from collections import Counter

ARCHIVE = r"C:\Users\leloc\Downloads\iot_23_datasets_small.tar.gz"

print("=" * 70)
print("IoT-23 DATASET INSPECTION")
print("=" * 70)

with tarfile.open(ARCHIVE, "r:gz") as tar:

    files = [
        m for m in tar.getmembers()
        if m.name.endswith("conn.log.labeled")
    ]

    print(f"\nFound {len(files)} conn.log.labeled files")

    total = Counter()

    for member in files:
        print(f"\n===== {member.name} =====")

        counter = Counter()

        f = tar.extractfile(member)

        if f is None:
            continue

        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")

            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")

            if len(fields) < 22:
                continue

            label = fields[20].strip()
            detailed_label = fields[21].strip()

            if label == "Benign":
                counter["Benign"] += 1

            elif detailed_label == "DDoS":
                counter["DDoS"] += 1

            elif detailed_label == "PartOfAHorizontalPortScan":
                counter["PartOfAHorizontalPortScan"] += 1

        for name, count in counter.most_common():
            print(f"  {name:35s} {count:,}")
            total[name] += count

    print("\n" + "=" * 70)
    print("TOTAL")
    print("=" * 70)

    for name, count in total.most_common():
        print(f"{name:35s} {count:,}")
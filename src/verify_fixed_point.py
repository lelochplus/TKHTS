import json
from pathlib import Path

import numpy as np


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

QUANT_DIR = ROOT / "models" / "quantized"
GOLDEN_DIR = ROOT / "results" / "golden"


# ============================================================
# LOAD DATA
# ============================================================
def load_mem(path, dtype=np.int32):
    """Đọc file .mem dạng HEX và chuyển về số signed."""
    values = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            value = int(line, 16)

            # INT8 signed
            if dtype == np.int8:
                if value >= 0x80:
                    value -= 0x100

            # INT32 signed
            elif dtype == np.int32:
                if value >= 0x80000000:
                    value -= 0x100000000

            values.append(value)

    return np.array(values, dtype=dtype)
def load_txt_matrix(path):
    """Đọc file golden dạng ma trận, mỗi dòng là một vector."""
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append([int(x) for x in line.split()])

    return np.array(rows, dtype=np.int32)


# ============================================================
# LOAD WEIGHTS / BIAS
# ============================================================

print("=" * 60)
print("FIXED-POINT VERIFICATION")
print("=" * 60)

print("\nLoading weights and biases...")

W1 = load_mem(QUANT_DIR / "W1_int8.mem", np.int8).reshape(32, 17)
W2 = load_mem(QUANT_DIR / "W2_int8.mem", np.int8).reshape(16, 32)
W3 = load_mem(QUANT_DIR / "W3_int8.mem", np.int8).reshape(3, 16)

b1 = load_mem(QUANT_DIR / "b1_int32.mem", np.int32)
b2 = load_mem(QUANT_DIR / "b2_int32.mem", np.int32)
b3 = load_mem(QUANT_DIR / "b3_int32.mem", np.int32)

print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")
print(f"W3 shape: {W3.shape}")

print(f"b1 shape: {b1.shape}")
print(f"b2 shape: {b2.shape}")
print(f"b3 shape: {b3.shape}")


# ============================================================
# LOAD FIXED-POINT PARAMETERS
# ============================================================

fixed_path = QUANT_DIR / "fixed_point_params.json"

if not fixed_path.exists():
    print("\nERROR: fixed_point_params.json not found!")
    print(f"Expected: {fixed_path}")
    raise SystemExit(1)

with open(fixed_path, "r") as f:
    fixed = json.load(f)


# File được tạo từ script generate_fixed_point_params.py
SHIFT = int(fixed["shift"])
M1 = int(fixed["h1"]["multiplier"])
M2 = int(fixed["h2"]["multiplier"])
ROUND = 1 << (SHIFT - 1)

print("\nFixed-point parameters:")
print(f"M1    = {M1}")
print(f"M2    = {M2}")
print(f"SHIFT = {SHIFT}")


# ============================================================
# LOAD GOLDEN VECTORS
# ============================================================

print("\nLoading golden vectors...")

X = load_txt_matrix(GOLDEN_DIR / "input_int8.txt")
GOLD_H1 = load_txt_matrix(GOLDEN_DIR / "expected_h1_int8.txt")
GOLD_H2 = load_txt_matrix(GOLDEN_DIR / "expected_h2_int8.txt")
GOLD_ACC3 = load_txt_matrix(GOLDEN_DIR / "expected_acc3_int32.txt")

with open(GOLDEN_DIR / "expected_class.txt", "r") as f:
    GOLD_CLASS = np.array(
        [int(line.strip()) for line in f if line.strip()],
        dtype=np.int32
    )

print(f"Samples = {len(X)}")


# ============================================================
# CHECK DIMENSIONS
# ============================================================

if X.shape[1] != 17:
    raise ValueError(f"Input dimension is {X.shape[1]}, expected 17")

if G0 := False:
    pass

if len(X) != len(GOLD_H1):
    raise ValueError("Input and H1 golden vector count mismatch")

if len(X) != len(GOLD_H2):
    raise ValueError("Input and H2 golden vector count mismatch")

if len(X) != len(GOLD_ACC3):
    raise ValueError("Input and ACC3 golden vector count mismatch")

if len(X) != len(GOLD_CLASS):
    raise ValueError("Input and class golden vector count mismatch")


# ============================================================
# FIXED-POINT ROUNDING
# ============================================================

def fixed_round_shift(value, multiplier, shift):
    """
    Fixed-point:
        y = round(value * multiplier / 2^shift)

    Hardware form:
        (value * multiplier + 2^(shift-1)) >> shift
    """

    value = np.asarray(value, dtype=np.int64)

    result = (value * multiplier + (1 << (shift - 1))) >> shift

    return result


# ============================================================
# VERIFY
# ============================================================

h1_errors = 0
h2_errors = 0
acc3_errors = 0
class_errors = 0

first_h1_error = None
first_h2_error = None
first_acc3_error = None
first_class_error = None


print("\nRunning verification...")


for i in range(len(X)):

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    x = X[i].astype(np.int64)

    # --------------------------------------------------------
    # LAYER 1
    #
    # ACC1 = W1 * X + b1
    # ReLU
    # Fixed-point conversion
    # --------------------------------------------------------

    acc1 = b1.astype(np.int64) + W1.astype(np.int64) @ x

    acc1_relu = np.maximum(acc1, 0)

    h1 = fixed_round_shift(
        acc1_relu,
        M1,
        SHIFT
    )

    # INT8/7-bit activation range used by hardware
    h1 = np.clip(h1, 0, 127)

    # --------------------------------------------------------
    # Compare H1
    # --------------------------------------------------------

    if not np.array_equal(h1.astype(np.int32), GOLD_H1[i]):

        h1_errors += 1

        if first_h1_error is None:
            first_h1_error = (
                i,
                h1.copy(),
                GOLD_H1[i].copy()
            )

    # --------------------------------------------------------
    # LAYER 2
    #
    # ACC2 = W2 * H1 + b2
    # ReLU
    # Fixed-point conversion
    # --------------------------------------------------------

    acc2 = b2.astype(np.int64) + W2.astype(np.int64) @ h1

    acc2_relu = np.maximum(acc2, 0)

    h2 = fixed_round_shift(
        acc2_relu,
        M2,
        SHIFT
    )

    h2 = np.clip(h2, 0, 127)

    # --------------------------------------------------------
    # Compare H2
    # --------------------------------------------------------

    if not np.array_equal(h2.astype(np.int32), GOLD_H2[i]):

        h2_errors += 1

        if first_h2_error is None:
            first_h2_error = (
                i,
                h2.copy(),
                GOLD_H2[i].copy()
            )

    # --------------------------------------------------------
    # LAYER 3
    #
    # ACC3 = W3 * H2 + b3
    #
    # Không quantize tiếp.
    # --------------------------------------------------------

    acc3 = b3.astype(np.int64) + W3.astype(np.int64) @ h2

    # --------------------------------------------------------
    # Compare ACC3
    # --------------------------------------------------------

    if not np.array_equal(acc3.astype(np.int32), GOLD_ACC3[i]):

        acc3_errors += 1

        if first_acc3_error is None:
            first_acc3_error = (
                i,
                acc3.copy(),
                GOLD_ACC3[i].copy()
            )

    # --------------------------------------------------------
    # CLASS
    # --------------------------------------------------------

    predicted_class = int(np.argmax(acc3))

    if predicted_class != int(GOLD_CLASS[i]):

        class_errors += 1

        if first_class_error is None:
            first_class_error = (
                i,
                predicted_class,
                int(GOLD_CLASS[i])
            )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"\nH1 errors    : {h1_errors}")
print(f"H2 errors    : {h2_errors}")
print(f"ACC3 errors  : {acc3_errors}")
print(f"CLASS errors : {class_errors}")


# ============================================================
# SHOW FIRST ERRORS
# ============================================================

if first_h1_error is not None:

    i, actual, expected = first_h1_error

    print("\nFirst H1 mismatch:")
    print(f"Sample  : {i}")
    print(f"Actual  : {actual.tolist()}")
    print(f"Expected: {expected.tolist()}")


if first_h2_error is not None:

    i, actual, expected = first_h2_error

    print("\nFirst H2 mismatch:")
    print(f"Sample  : {i}")
    print(f"Actual  : {actual.tolist()}")
    print(f"Expected: {expected.tolist()}")


if first_acc3_error is not None:

    i, actual, expected = first_acc3_error

    print("\nFirst ACC3 mismatch:")
    print(f"Sample  : {i}")
    print(f"Actual  : {actual.tolist()}")
    print(f"Expected: {expected.tolist()}")


if first_class_error is not None:

    i, actual, expected = first_class_error

    print("\nFirst CLASS mismatch:")
    print(f"Sample  : {i}")
    print(f"Actual  : {actual}")
    print(f"Expected: {expected}")


# ============================================================
# FINAL STATUS
# ============================================================

if (
    h1_errors == 0
    and h2_errors == 0
    and acc3_errors == 0
    and class_errors == 0
):

    print("\n" + "=" * 60)
    print("FIXED-POINT VERIFICATION: PASS")
    print("=" * 60)

else:

    print("\n" + "=" * 60)
    print("FIXED-POINT VERIFICATION: FAIL")
    print("=" * 60)

    print("\nCần kiểm tra:")
    print("1. M1 / M2 / SHIFT")
    print("2. Rounding")
    print("3. ReLU")
    print("4. Weight / bias")
    print("5. Thứ tự feature")
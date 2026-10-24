import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "iot23_behavior_test.csv"
)

QUANT_DIR = (
    BASE_DIR
    / "models"
    / "quantized"
)

OUTPUT_DIR = BASE_DIR / "results" / "golden"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

NUM_FEATURES = 17

# Số mẫu golden
NUM_SAMPLES = 1000

RANDOM_SEED = 42


# ============================================================
# LOAD QUANTIZATION PARAMETERS
# ============================================================

print("=" * 70)
print("GENERATE GOLDEN TEST VECTORS")
print("=" * 70)

print()
print("Loading quantization parameters...")

with open(
    QUANT_DIR / "quantization.json",
    "r"
) as f:
    qinfo = json.load(f)


S_X = qinfo["activation_scales"]["input"]
S_H1 = qinfo["activation_scales"]["H1"]
S_H2 = qinfo["activation_scales"]["H2"]

S_W1 = qinfo["weight_scales"]["W1"]
S_W2 = qinfo["weight_scales"]["W2"]
S_W3 = qinfo["weight_scales"]["W3"]

print()
print("Activation scales:")
print(f"    S_X  = {S_X}")
print(f"    S_H1 = {S_H1}")
print(f"    S_H2 = {S_H2}")

print()
print("Weight scales:")
print(f"    S_W1 = {S_W1}")
print(f"    S_W2 = {S_W2}")
print(f"    S_W3 = {S_W3}")


# ============================================================
# LOAD WEIGHTS
# ============================================================

print()
print("Loading INT8 weights...")

W1 = np.load(
    QUANT_DIR / "W1_int8.npy"
)

W2 = np.load(
    QUANT_DIR / "W2_int8.npy"
)

W3 = np.load(
    QUANT_DIR / "W3_int8.npy"
)

b1 = np.load(
    QUANT_DIR / "b1_int32.npy"
)

b2 = np.load(
    QUANT_DIR / "b2_int32.npy"
)

b3 = np.load(
    QUANT_DIR / "b3_int32.npy"
)

print()
print("Shapes:")

print(f"    W1 = {W1.shape}")
print(f"    b1 = {b1.shape}")

print(f"    W2 = {W2.shape}")
print(f"    b2 = {b2.shape}")

print(f"    W3 = {W3.shape}")
print(f"    b3 = {b3.shape}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("Loading test dataset...")

df = pd.read_csv(TEST_PATH)

X = df.iloc[:, :NUM_FEATURES].values.astype(
    np.float32
)

y = df.iloc[:, NUM_FEATURES].values.astype(
    np.int32
)

print()
print(f"Total test samples: {len(X):,}")


# ============================================================
# SELECT SAMPLES
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

indices = rng.choice(
    len(X),
    size=NUM_SAMPLES,
    replace=False
)

indices.sort()

X = X[indices]
y = y[indices]


# ============================================================
# INPUT QUANTIZATION
# ============================================================

print()
print("Quantizing input...")

X_q = np.round(
    X / S_X
)

X_q = np.clip(
    X_q,
    -127,
    127
).astype(np.int8)


# ============================================================
# LAYER 1
# ============================================================

print()
print("Calculating Layer 1...")

acc1 = (
    X_q.astype(np.int32)
    @ W1.T.astype(np.int32)
)

acc1 += b1

# ReLU
acc1_relu = np.maximum(
    acc1,
    0
)


# Requantize H1

H1_q = np.round(
    acc1_relu.astype(np.float32)
    * (S_X * S_W1)
    / S_H1
)

H1_q = np.clip(
    H1_q,
    0,
    127
).astype(np.int8)


# ============================================================
# LAYER 2
# ============================================================

print("Calculating Layer 2...")

acc2 = (
    H1_q.astype(np.int32)
    @ W2.T.astype(np.int32)
)

acc2 += b2

# ReLU
acc2_relu = np.maximum(
    acc2,
    0
)


# Requantize H2

H2_q = np.round(
    acc2_relu.astype(np.float32)
    * (S_H1 * S_W2)
    / S_H2
)

H2_q = np.clip(
    H2_q,
    0,
    127
).astype(np.int8)


# ============================================================
# LAYER 3
# ============================================================

print("Calculating Layer 3...")

acc3 = (
    H2_q.astype(np.int32)
    @ W3.T.astype(np.int32)
)

acc3 += b3


# ============================================================
# PREDICTION
# ============================================================

pred = np.argmax(
    acc3,
    axis=1
).astype(np.int32)


# ============================================================
# CHECK
# ============================================================

accuracy = np.mean(
    pred == y
)

print()
print("=" * 70)
print("GOLDEN VECTOR CHECK")
print("=" * 70)

print()
print(f"Samples  : {NUM_SAMPLES:,}")
print(f"Correct  : {(pred == y).sum():,}")
print(f"Accuracy : {accuracy * 100:.4f}%")


# ============================================================
# SAVE INPUT VECTORS
#
# Each line:
#
# x0 x1 ... x16
# ============================================================

input_path = (
    OUTPUT_DIR
    / "input_int8.txt"
)

with open(
    input_path,
    "w"
) as f:

    for row in X_q:

        f.write(
            " ".join(
                str(int(v))
                for v in row
            )
            + "\n"
        )


# ============================================================
# SAVE EXPECTED OUTPUT
# ============================================================

output_path = (
    OUTPUT_DIR
    / "expected_class.txt"
)

with open(
    output_path,
    "w"
) as f:

    for value in pred:

        f.write(
            f"{int(value)}\n"
        )


# ============================================================
# SAVE INTERMEDIATE RESULTS
#
# Useful for debugging Verilog
# ============================================================

h1_path = (
    OUTPUT_DIR
    / "expected_h1_int8.txt"
)

with open(
    h1_path,
    "w"
) as f:

    for row in H1_q:

        f.write(
            " ".join(
                str(int(v))
                for v in row
            )
            + "\n"
        )


h2_path = (
    OUTPUT_DIR
    / "expected_h2_int8.txt"
)

with open(
    h2_path,
    "w"
) as f:

    for row in H2_q:

        f.write(
            " ".join(
                str(int(v))
                for v in row
            )
            + "\n"
        )


# ============================================================
# SAVE OUTPUT ACCUMULATOR
# ============================================================

acc3_path = (
    OUTPUT_DIR
    / "expected_acc3_int32.txt"
)

with open(
    acc3_path,
    "w"
) as f:

    for row in acc3:

        f.write(
            " ".join(
                str(int(v))
                for v in row
            )
            + "\n"
        )


# ============================================================
# SAVE LABELS
# ============================================================

label_path = (
    OUTPUT_DIR
    / "expected_label.txt"
)

with open(
    label_path,
    "w"
) as f:

    for value in y:

        f.write(
            f"{int(value)}\n"
        )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "num_samples": NUM_SAMPLES,

    "input_dimension": 17,

    "architecture": [
        17,
        32,
        16,
        3
    ],

    "activation_scales": {
        "input": S_X,
        "H1": S_H1,
        "H2": S_H2
    },

    "weight_scales": {
        "W1": S_W1,
        "W2": S_W2,
        "W3": S_W3
    },

    "accuracy": float(accuracy),

    "files": {
        "input": "input_int8.txt",
        "expected_h1": "expected_h1_int8.txt",
        "expected_h2": "expected_h2_int8.txt",
        "expected_acc3": "expected_acc3_int32.txt",
        "expected_class": "expected_class.txt",
        "expected_label": "expected_label.txt"
    }
}


with open(
    OUTPUT_DIR / "metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# DISPLAY FIRST SAMPLE
# ============================================================

print()
print("=" * 70)
print("FIRST GOLDEN SAMPLE")
print("=" * 70)

print()
print("Input INT8:")
print(
    X_q[0].tolist()
)

print()
print("H1 INT8:")
print(
    H1_q[0].tolist()
)

print()
print("H2 INT8:")
print(
    H2_q[0].tolist()
)

print()
print("ACC3:")
print(
    acc3[0].tolist()
)

print()
print(f"Expected label : {y[0]}")
print(f"Predicted class: {pred[0]}")


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("GOLDEN VECTOR GENERATION COMPLETE")
print("=" * 70)

print()
print("Output directory:")
print(
    OUTPUT_DIR
)

print()
print("Generated files:")

print("    input_int8.txt")
print("    expected_h1_int8.txt")
print("    expected_h2_int8.txt")
print("    expected_acc3_int32.txt")
print("    expected_class.txt")
print("    expected_label.txt")
print("    metadata.json")
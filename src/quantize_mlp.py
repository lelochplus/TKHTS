import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mlp_behavior_17_32_16_3.pth"
SCALER_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "standard_scaler_behavior.joblib"
)

VAL_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "iot23_behavior_val.csv"
)

TEST_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "iot23_behavior_test.csv"
)

OUTPUT_DIR = BASE_DIR / "models" / "quantized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Model
INPUT_SIZE = 17
HIDDEN1 = 32
HIDDEN2 = 16
NUM_CLASSES = 3

# Calibration
CALIBRATION_SAMPLES = 100_000

# Thay vì max tuyệt đối.
# Có thể thử 99.9 hoặc 99.99 nếu cần.
CALIBRATION_PERCENTILE = 99.9

# Test batch
BATCH_SIZE = 4096


# ============================================================
# MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("IoT-23 MLP INT8 QUANTIZATION - PERCENTILE PTQ")
print("=" * 70)

print()
print("Architecture:")
print(f"    {INPUT_SIZE} -> {HIDDEN1} -> {HIDDEN2} -> {NUM_CLASSES}")

print()
print("Loading model:")
print(f"    {MODEL_PATH}")

model = MLP()

state = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)
model.eval()

print("    Model loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

print()
print("Loading StandardScaler:")
print(f"    {SCALER_PATH}")

scaler = joblib.load(SCALER_PATH)

print("    Scaler loaded successfully.")


# ============================================================
# EXTRACT FLOAT WEIGHTS
# ============================================================

W1 = model.network[0].weight.detach().numpy().astype(np.float32)
b1 = model.network[0].bias.detach().numpy().astype(np.float32)

W2 = model.network[2].weight.detach().numpy().astype(np.float32)
b2 = model.network[2].bias.detach().numpy().astype(np.float32)

W3 = model.network[4].weight.detach().numpy().astype(np.float32)
b3 = model.network[4].bias.detach().numpy().astype(np.float32)


print()
print("=" * 70)
print("MODEL STRUCTURE CHECK")
print("=" * 70)

print(f"    W1 : {W1.shape}")
print(f"    b1 : {b1.shape}")
print(f"    W2 : {W2.shape}")
print(f"    b2 : {b2.shape}")
print(f"    W3 : {W3.shape}")
print(f"    b3 : {b3.shape}")


# ============================================================
# WEIGHT QUANTIZATION
# Symmetric INT8
# ============================================================

def quantize_weight(weight):

    max_abs = np.max(np.abs(weight))

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0

    q = np.round(weight / scale)
    q = np.clip(q, -127, 127)

    return q.astype(np.int8), float(scale)


print()
print("=" * 70)
print("WEIGHT QUANTIZATION")
print("=" * 70)

W1_q, S_W1 = quantize_weight(W1)
W2_q, S_W2 = quantize_weight(W2)
W3_q, S_W3 = quantize_weight(W3)

print()
print("Weight scales:")
print(f"    W1 = {S_W1}")
print(f"    W2 = {S_W2}")
print(f"    W3 = {S_W3}")

print()
print("INT8 ranges:")
print(f"    W1 = {W1_q.min()} ... {W1_q.max()}")
print(f"    W2 = {W2_q.min()} ... {W2_q.max()}")
print(f"    W3 = {W3_q.min()} ... {W3_q.max()}")


# ============================================================
# CALIBRATION
#
# IMPORTANT:
# Use EXACTLY 100,000 samples.
#
# Collect:
#   Input
#   H1
#   H2
#
# Then calculate percentile of ABSOLUTE values.
# ============================================================

print()
print("=" * 70)
print("ACTIVATION CALIBRATION")
print("=" * 70)

print()
print("Calibration dataset:")
print(f"    {VAL_PATH}")

print()
print(f"Calibration samples: {CALIBRATION_SAMPLES}")
print(f"Percentile        : {CALIBRATION_PERCENTILE}%")


# ------------------------------------------------------------
# Read validation data
# ------------------------------------------------------------

df_cal = pd.read_csv(VAL_PATH)

X_cal = df_cal.iloc[:, :INPUT_SIZE].values.astype(np.float32)

# Shuffle deterministic
rng = np.random.default_rng(42)

indices = rng.permutation(len(X_cal))

if len(indices) < CALIBRATION_SAMPLES:
    raise RuntimeError(
        f"Validation dataset chỉ có {len(indices)} samples."
    )

indices = indices[:CALIBRATION_SAMPLES]

X_cal = X_cal[indices]

print()
print(f"Calibration loaded: {len(X_cal):,}")


# ------------------------------------------------------------
# FLOAT forward pass
# ------------------------------------------------------------

with torch.no_grad():

    X_tensor = torch.from_numpy(X_cal)

    H1 = model.network[0](X_tensor)
    H1 = torch.relu(H1)

    H2 = model.network[2](H1)
    H2 = torch.relu(H2)

    H1_np = H1.numpy()
    H2_np = H2.numpy()


# ------------------------------------------------------------
# Percentile calibration
# ------------------------------------------------------------

input_abs = np.abs(X_cal).reshape(-1)
h1_abs = np.abs(H1_np).reshape(-1)
h2_abs = np.abs(H2_np).reshape(-1)


input_max = np.percentile(
    input_abs,
    CALIBRATION_PERCENTILE
)

h1_max = np.percentile(
    h1_abs,
    CALIBRATION_PERCENTILE
)

h2_max = np.percentile(
    h2_abs,
    CALIBRATION_PERCENTILE
)


# Avoid zero
input_max = max(float(input_max), 1e-12)
h1_max = max(float(h1_max), 1e-12)
h2_max = max(float(h2_max), 1e-12)


# ------------------------------------------------------------
# Activation scales
# ------------------------------------------------------------

S_X = input_max / 127.0
S_H1 = h1_max / 127.0
S_H2 = h2_max / 127.0


print()
print("Calibration result:")
print(f"    Samples       : {len(X_cal):,}")
print(f"    Input {CALIBRATION_PERCENTILE}% : {input_max}")
print(f"    H1    {CALIBRATION_PERCENTILE}% : {h1_max}")
print(f"    H2    {CALIBRATION_PERCENTILE}% : {h2_max}")

print()
print("Activation scales:")
print(f"    Input = {S_X}")
print(f"    H1    = {S_H1}")
print(f"    H2    = {S_H2}")


# ============================================================
# BIAS QUANTIZATION
#
# b1:
#     input scale * weight scale
#
# b2:
#     H1 scale * W2 scale
#
# b3:
#     H2 scale * W3 scale
# ============================================================

S_B1 = S_X * S_W1
S_B2 = S_H1 * S_W2
S_B3 = S_H2 * S_W3


b1_q = np.round(b1 / S_B1).astype(np.int32)
b2_q = np.round(b2 / S_B2).astype(np.int32)
b3_q = np.round(b3 / S_B3).astype(np.int32)


print()
print("=" * 70)
print("BIAS QUANTIZATION")
print("=" * 70)

print()
print("Bias scales:")
print(f"    b1 = {S_B1}")
print(f"    b2 = {S_B2}")
print(f"    b3 = {S_B3}")


# ============================================================
# INT8 INFERENCE
# ============================================================

def int8_predict(X):

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    X_q = np.round(X / S_X)

    X_q = np.clip(
        X_q,
        -127,
        127
    ).astype(np.int8)


    # --------------------------------------------------------
    # LAYER 1
    # --------------------------------------------------------

    acc1 = (
        X_q.astype(np.int32) @
        W1_q.T.astype(np.int32)
    )

    acc1 += b1_q

    # ReLU
    acc1 = np.maximum(acc1, 0)


    # --------------------------------------------------------
    # REQUANTIZE LAYER 1
    #
    # float H1 ≈ int_H1 * S_H1
    #
    # acc1 scale:
    #
    #     S_X * S_W1
    #
    # --------------------------------------------------------

    H1_q = np.round(
        acc1.astype(np.float32) *
        (S_X * S_W1) /
        S_H1
    )

    H1_q = np.clip(
        H1_q,
        0,
        127
    ).astype(np.int8)


    # --------------------------------------------------------
    # LAYER 2
    # --------------------------------------------------------

    acc2 = (
        H1_q.astype(np.int32) @
        W2_q.T.astype(np.int32)
    )

    acc2 += b2_q

    acc2 = np.maximum(acc2, 0)


    # --------------------------------------------------------
    # REQUANTIZE LAYER 2
    # --------------------------------------------------------

    H2_q = np.round(
        acc2.astype(np.float32) *
        (S_H1 * S_W2) /
        S_H2
    )

    H2_q = np.clip(
        H2_q,
        0,
        127
    ).astype(np.int8)


    # --------------------------------------------------------
    # LAYER 3
    # --------------------------------------------------------

    acc3 = (
        H2_q.astype(np.int32) @
        W3_q.T.astype(np.int32)
    )

    acc3 += b3_q

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # All classes have the same output scale:
    #
    #     S_H2 * S_W3
    #
    # Therefore argmax(acc3) is equivalent to argmax(logits).
    # --------------------------------------------------------

    pred = np.argmax(
        acc3,
        axis=1
    )

    return pred


# ============================================================
# FLOAT TEST
# ============================================================

print()
print("=" * 70)
print("FLOAT MODEL SANITY CHECK")
print("=" * 70)

df_test = pd.read_csv(TEST_PATH)

X_test = df_test.iloc[:, :INPUT_SIZE].values.astype(np.float32)
y_test = df_test.iloc[:, INPUT_SIZE].values.astype(np.int64)

print()
print(f"Test samples: {len(X_test):,}")

with torch.no_grad():

    float_logits = model(
        torch.from_numpy(X_test)
    )

    float_pred = torch.argmax(
        float_logits,
        dim=1
    ).numpy()

float_acc = accuracy_score(
    y_test,
    float_pred
)

print()
print(f"Float Accuracy: {float_acc:.6f}")
print(f"Float Accuracy: {float_acc * 100:.4f}%")


# ============================================================
# INT8 TEST
# ============================================================

print()
print("=" * 70)
print("INT8 TEST")
print("=" * 70)

all_pred = []

total = len(X_test)

for start in range(0, total, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        total
    )

    X_batch = X_test[start:end]

    pred = int8_predict(
        X_batch
    )

    all_pred.append(pred)

    if end % 40960 == 0 or end == total:
        print(
            f"Test: {end:,} / {total:,}"
        )


int8_pred = np.concatenate(
    all_pred
)


# ============================================================
# RESULTS
# ============================================================

int8_acc = accuracy_score(
    y_test,
    int8_pred
)

cm = confusion_matrix(
    y_test,
    int8_pred
)

print()
print("=" * 70)
print("INT8 RESULT")
print("=" * 70)

print()
print(f"Correct : {(int8_pred == y_test).sum():,}")
print(f"Total   : {len(y_test):,}")

print()
print(f"Accuracy: {int8_acc:.6f}")
print(f"Accuracy: {int8_acc * 100:.4f}%")

print()
print("INT8 Confusion Matrix:")
print(cm)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        int8_pred,
        target_names=[
            "Normal",
            "DoS",
            "Port Scan"
        ],
        digits=4
    )
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("=" * 70)
print("FLOAT vs INT8")
print("=" * 70)

print()
print(f"Float Accuracy : {float_acc * 100:.4f}%")
print(f"INT8 Accuracy  : {int8_acc * 100:.4f}%")

print()
print(
    f"Accuracy Loss  : "
    f"{(float_acc - int8_acc) * 100:.4f}%"
)


# ============================================================
# EXPORT
# ============================================================

print()
print("=" * 70)
print("EXPORT FOR VERILOG")
print("=" * 70)

print()
print(f"Export directory:")
print(f"    {OUTPUT_DIR}")


# ------------------------------------------------------------
# Save MEM files
# ------------------------------------------------------------

def save_mem_int8(path, array):

    flat = array.flatten()

    with open(path, "w") as f:

        for value in flat:

            value = int(value)

            # Convert signed INT8 to 8-bit hex
            value &= 0xFF

            f.write(
                f"{value:02X}\n"
            )


def save_mem_int32(path, array):

    flat = array.flatten()

    with open(path, "w") as f:

        for value in flat:

            value = int(value)

            # Convert signed INT32 to 32-bit hex
            value &= 0xFFFFFFFF

            f.write(
                f"{value:08X}\n"
            )


save_mem_int8(
    OUTPUT_DIR / "W1_int8.mem",
    W1_q
)

save_mem_int32(
    OUTPUT_DIR / "b1_int32.mem",
    b1_q
)

save_mem_int8(
    OUTPUT_DIR / "W2_int8.mem",
    W2_q
)

save_mem_int32(
    OUTPUT_DIR / "b2_int32.mem",
    b2_q
)

save_mem_int8(
    OUTPUT_DIR / "W3_int8.mem",
    W3_q
)

save_mem_int32(
    OUTPUT_DIR / "b3_int32.mem",
    b3_q
)


# ============================================================
# SAVE NUMPY
# ============================================================

np.save(
    OUTPUT_DIR / "W1_int8.npy",
    W1_q
)

np.save(
    OUTPUT_DIR / "b1_int32.npy",
    b1_q
)

np.save(
    OUTPUT_DIR / "W2_int8.npy",
    W2_q
)

np.save(
    OUTPUT_DIR / "b2_int32.npy",
    b2_q
)

np.save(
    OUTPUT_DIR / "W3_int8.npy",
    W3_q
)

np.save(
    OUTPUT_DIR / "b3_int32.npy",
    b3_q
)


# ============================================================
# SAVE QUANTIZATION PARAMETERS
# ============================================================

quantization_info = {

    "architecture": [
        INPUT_SIZE,
        HIDDEN1,
        HIDDEN2,
        NUM_CLASSES
    ],

    "calibration_samples":
        CALIBRATION_SAMPLES,

    "calibration_percentile":
        CALIBRATION_PERCENTILE,

    "weight_scales": {
        "W1": S_W1,
        "W2": S_W2,
        "W3": S_W3
    },

    "activation_scales": {
        "input": S_X,
        "H1": S_H1,
        "H2": S_H2
    },

    "bias_scales": {
        "b1": S_B1,
        "b2": S_B2,
        "b3": S_B3
    },

    "float_accuracy":
        float(float_acc),

    "int8_accuracy":
        float(int8_acc),

    "accuracy_loss":
        float(float_acc - int8_acc)
}


with open(
    OUTPUT_DIR / "quantization.json",
    "w"
) as f:

    json.dump(
        quantization_info,
        f,
        indent=4
    )


# ============================================================
# NORMALIZATION INFORMATION
# ============================================================

normalization_info = {

    "method":
        "StandardScaler",

    "scaler":
        str(SCALER_PATH),

    "feature_count":
        INPUT_SIZE,

    "features": [
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
        "s0_ratio_10s"
    ]
}


with open(
    OUTPUT_DIR / "normalization.json",
    "w"
) as f:

    json.dump(
        normalization_info,
        f,
        indent=4
    )


# ============================================================
# RESULT JSON
# ============================================================

result_info = {

    "float_accuracy":
        float(float_acc),

    "int8_accuracy":
        float(int8_acc),

    "accuracy_loss":
        float(float_acc - int8_acc),

    "confusion_matrix":
        cm.tolist(),

    "calibration_samples":
        CALIBRATION_SAMPLES,

    "calibration_percentile":
        CALIBRATION_PERCENTILE
}


with open(
    OUTPUT_DIR / "quantization_result.json",
    "w"
) as f:

    json.dump(
        result_info,
        f,
        indent=4
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("INT8 QUANTIZATION COMPLETE")
print("=" * 70)

print()
print(
    f"Float model accuracy : "
    f"{float_acc * 100:.4f}%"
)

print(
    f"INT8 model accuracy  : "
    f"{int8_acc * 100:.4f}%"
)

print()
print("Verilog weight files:")

print("    W1_int8.mem")
print("    b1_int32.mem")
print("    W2_int8.mem")
print("    b2_int32.mem")
print("    W3_int8.mem")
print("    b3_int32.mem")

print()
print("Parameter files:")

print("    normalization.json")
print("    quantization.json")

print()
print("=" * 70)

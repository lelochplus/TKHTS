from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(r"D:\Part4_Data_ESP32")

TRAIN_FILE = BASE_DIR / "dataset" / "processed" / "iot23_train.csv"
VAL_FILE   = BASE_DIR / "dataset" / "processed" / "iot23_val.csv"
TEST_FILE  = BASE_DIR / "dataset" / "processed" / "iot23_test.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model mới - không ghi đè model 11-16-8-3
MODEL_FILE = MODEL_DIR / "mlp_11_32_16_3.pth"


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 11

# MLP mới:
# 11 input -> 32 neuron -> 16 neuron -> 3 output
HIDDEN1 = 32
HIDDEN2 = 16

NUM_CLASSES = 3

BATCH_SIZE = 4096
EPOCHS = 15
LEARNING_RATE = 0.001

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

LABEL_INDEX = 11


# ============================================================
# DATASET
# ============================================================

class CSVIterableDataset:

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):

        with open(self.file_path, "r", encoding="utf-8") as f:

            # Bỏ dòng header
            next(f)

            features = []
            labels = []

            for line in f:

                values = line.strip().split(",")

                # Dataset phải có 12 cột:
                # 11 features + 1 class
                if len(values) != 12:
                    continue

                try:
                    x = [
                        float(values[i])
                        for i in range(INPUT_SIZE)
                    ]

                    y = int(values[LABEL_INDEX])

                except ValueError:
                    continue

                features.append(x)
                labels.append(y)

                # Đủ một batch
                if len(features) >= BATCH_SIZE:

                    yield (
                        torch.tensor(
                            features,
                            dtype=torch.float32
                        ),

                        torch.tensor(
                            labels,
                            dtype=torch.long
                        )
                    )

                    features = []
                    labels = []

            # Batch cuối
            if features:

                yield (
                    torch.tensor(
                        features,
                        dtype=torch.float32
                    ),

                    torch.tensor(
                        labels,
                        dtype=torch.long
                    )
                )


# ============================================================
# MLP MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Layer 1
            nn.Linear(
                INPUT_SIZE,
                HIDDEN1
            ),

            nn.ReLU(),

            # Layer 2
            nn.Linear(
                HIDDEN1,
                HIDDEN2
            ),

            nn.ReLU(),

            # Output layer
            nn.Linear(
                HIDDEN2,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, file_path, device):

    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for x, y in CSVIterableDataset(file_path):

            x = x.to(device)
            y = y.to(device)

            # Forward
            output = model(x)

            # Chọn class có logit lớn nhất
            predictions = torch.argmax(
                output,
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    accuracy = correct / total

    return (
        accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 60)
    print("IoT-23 MLP TRAINING")
    print("=" * 60)

    # Chọn CPU hoặc GPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device      : {device}")

    print(
        f"Architecture: "
        f"{INPUT_SIZE} -> "
        f"{HIDDEN1} -> "
        f"{HIDDEN2} -> "
        f"{NUM_CLASSES}"
    )

    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Epochs      : {EPOCHS}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MLP().to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_samples = 0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        # Đọc tập train theo batch
        for x, y in CSVIterableDataset(
            TRAIN_FILE
        ):

            x = x.to(device)
            y = y.to(device)

            # Xóa gradient cũ
            optimizer.zero_grad()

            # Forward
            output = model(x)

            # Tính loss
            loss = criterion(
                output,
                y
            )

            # Backpropagation
            loss.backward()

            # Cập nhật weight
            optimizer.step()

            batch_size = y.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        # Loss trung bình
        avg_loss = (
            total_loss /
            total_samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_acc, _, _ = evaluate(
            model,
            VAL_FILE,
            device
        )

        print(
            f"    Loss      : "
            f"{avg_loss:.6f}"
        )

        print(
            f"    Val Acc   : "
            f"{val_acc:.6f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                MODEL_FILE
            )

            print(
                "    Best model saved"
            )

    # ========================================================
    # TESTING
    # ========================================================

    print("\n" + "=" * 60)
    print("TESTING BEST MODEL")
    print("=" * 60)

    # Load model tốt nhất
    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device
        )
    )

    # Test
    test_acc, y_true, y_pred = evaluate(
        model,
        TEST_FILE,
        device
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.6f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,

            target_names=[
                "Normal",
                "DoS",
                "Port Scan"
            ],

            digits=4
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(cm)

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print(
        f"\nModel saved: "
        f"{MODEL_FILE}"
    )

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()
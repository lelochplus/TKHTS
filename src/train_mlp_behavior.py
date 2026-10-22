from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(r"D:\Part4_Data_ESP32")

DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

TRAIN_FILE = DATA_DIR / "iot23_behavior_train.csv"
VAL_FILE = DATA_DIR / "iot23_behavior_val.csv"
TEST_FILE = DATA_DIR / "iot23_behavior_test.csv"

MODEL_FILE = MODEL_DIR / "mlp_behavior_17_32_16_3.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 17
HIDDEN1 = 32
HIDDEN2 = 16
NUM_CLASSES = 3

BATCH_SIZE = 4096
EPOCHS = 15
LEARNING_RATE = 0.001

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


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
# CSV BATCH LOADER
# ============================================================

def load_batches(filename, batch_size):

    features = []
    labels = []

    with open(
        filename,
        "r",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024
    ) as f:

        reader = csv.reader(f)

        # Skip header
        next(reader)

        for row in reader:

            x = [
                float(v)
                for v in row[:INPUT_SIZE]
            ]

            y = int(row[INPUT_SIZE])

            features.append(x)
            labels.append(y)

            if len(features) >= batch_size:

                yield (
                    torch.tensor(
                        np.asarray(
                            features,
                            dtype=np.float32
                        )
                    ),
                    torch.tensor(
                        labels,
                        dtype=torch.long
                    )
                )

                features = []
                labels = []

        if features:

            yield (
                torch.tensor(
                    np.asarray(
                        features,
                        dtype=np.float32
                    )
                ),
                torch.tensor(
                    labels,
                    dtype=torch.long
                )
            )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, filename, device):

    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for x, y in load_batches(
            filename,
            BATCH_SIZE
        ):

            x = x.to(device)
            y = y.to(device)

            output = model(x)

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
        np.asarray(all_labels),
        np.asarray(all_predictions)
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("IoT-23 BEHAVIOR MLP TRAINING")
print("=" * 70)

print()
print("Architecture:")
print("    17 → 32 → 16 → 3")

print()
print(f"Batch size    : {BATCH_SIZE}")
print(f"Epochs        : {EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(f"Device        : {device}")


# ============================================================
# CREATE MODEL
# ============================================================

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# PARAMETER COUNT
# ============================================================

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print()
print(f"Parameters    : {parameters}")


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

best_val_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in load_batches(
        TRAIN_FILE,
        BATCH_SIZE
    ):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        batch_size = y.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    train_loss = (
        total_loss
        / total_samples
    )

    val_accuracy, _, _ = evaluate(
        model,
        VAL_FILE,
        device
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.6f} | "
        f"Val Acc: {val_accuracy:.6f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.6f}"
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 70)
print("TEST")
print("=" * 70)

test_accuracy, y_true, y_pred = evaluate(
    model,
    TEST_FILE,
    device
)

print()
print(
    f"Test Accuracy: "
    f"{test_accuracy:.6f}"
)

print()
print("Classification Report:")

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

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE = (
    RESULT_DIR
    / "mlp_behavior_results.txt"
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "IoT-23 Behavior MLP\n"
    )

    f.write(
        "Architecture: "
        "17 -> 32 -> 16 -> 3\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.6f}\n\n"
    )

    f.write(
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

    f.write("\nConfusion Matrix:\n")

    f.write(
        str(
            confusion_matrix(
                y_true,
                y_pred
            )
        )
    )


print()
print("Model saved:")
print(f"    {MODEL_FILE}")

print()
print("Results saved:")
print(f"    {RESULT_FILE}")

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
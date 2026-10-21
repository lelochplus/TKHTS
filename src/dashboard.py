import tkinter as tk
from tkinter import ttk
import json
import threading
import time
import paho.mqtt.client as mqtt


# =====================================================
# MQTT CONFIG
# =====================================================

BROKER = "192.168.1.43"
PORT = 1883
TOPIC = "gateway/alert"


# =====================================================
# DATA
# =====================================================

count_normal = 0
count_dos = 0
count_portscan = 0

last_class = "-"
last_label = "Waiting..."
last_time = "--:--:--"


# =====================================================
# GUI UPDATE
# =====================================================

def update_dashboard(class_id, label):
    global count_normal
    global count_dos
    global count_portscan
    global last_class
    global last_label
    global last_time

    if class_id == 0:
        count_normal += 1
    elif class_id == 1:
        count_dos += 1
    elif class_id == 2:
        count_portscan += 1

    last_class = str(class_id)
    last_label = label
    last_time = time.strftime("%H:%M:%S")

    class_value.config(text=last_class)
    label_value.config(text=last_label)
    time_value.config(text=last_time)

    normal_value.config(text=str(count_normal))
    dos_value.config(text=str(count_dos))
    portscan_value.config(text=str(count_portscan))

    if class_id == 0:
        status_value.config(
            text="NORMAL",
            foreground="green"
        )

    elif class_id == 1:
        status_value.config(
            text="DOS ATTACK",
            foreground="red"
        )

    elif class_id == 2:
        status_value.config(
            text="PORT SCAN",
            foreground="orange"
        )


# =====================================================
# MQTT CALLBACK
# =====================================================

def on_connect(client, userdata, flags, rc):
    print("MQTT connected, rc =", rc)

    client.subscribe(TOPIC)

    print("Subscribed:", TOPIC)


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()

        print("MQTT RX:", payload)

        data = json.loads(payload)

        class_id = int(data["class"])
        label = data["label"]

        # GUI phai duoc update trong main thread
        root.after(
            0,
            update_dashboard,
            class_id,
            label
        )

    except Exception as e:
        print("Message error:", e)


# =====================================================
# MQTT THREAD
# =====================================================

def mqtt_thread():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION1,
        client_id="python-dashboard"
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting MQTT...")

    client.connect(
        BROKER,
        PORT,
        60
    )

    client.loop_forever()


# =====================================================
# RESET COUNTER
# =====================================================

def reset_counter():
    global count_normal
    global count_dos
    global count_portscan

    count_normal = 0
    count_dos = 0
    count_portscan = 0

    normal_value.config(text="0")
    dos_value.config(text="0")
    portscan_value.config(text="0")


# =====================================================
# GUI
# =====================================================

root = tk.Tk()

root.title("IoT Security Gateway")
root.geometry("650x500")

root.resizable(False, False)


# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

title = ttk.Label(
    root,
    text="IoT SECURITY GATEWAY",
    font=("Arial", 24, "bold")
)

title.pack(pady=20)


subtitle = ttk.Label(
    root,
    text="RISC-V + AI Intrusion Detection",
    font=("Arial", 12)
)

subtitle.pack()


# -----------------------------------------------------
# STATUS
# -----------------------------------------------------

status_frame = ttk.LabelFrame(
    root,
    text="Current Status"
)

status_frame.pack(
    padx=30,
    pady=20,
    fill="x"
)


status_value = ttk.Label(
    status_frame,
    text="WAITING",
    font=("Arial", 24, "bold")
)

status_value.pack(pady=15)


# -----------------------------------------------------
# CURRENT DATA
# -----------------------------------------------------

data_frame = ttk.Frame(root)

data_frame.pack(pady=10)


ttk.Label(
    data_frame,
    text="Class:",
    font=("Arial", 12)
).grid(row=0, column=0, padx=20, pady=5)

class_value = ttk.Label(
    data_frame,
    text="-",
    font=("Arial", 14, "bold")
)

class_value.grid(row=0, column=1, padx=20)


ttk.Label(
    data_frame,
    text="Label:",
    font=("Arial", 12)
).grid(row=1, column=0, padx=20, pady=5)

label_value = ttk.Label(
    data_frame,
    text="Waiting...",
    font=("Arial", 14, "bold")
)

label_value.grid(row=1, column=1, padx=20)


ttk.Label(
    data_frame,
    text="Last alert:",
    font=("Arial", 12)
).grid(row=2, column=0, padx=20, pady=5)

time_value = ttk.Label(
    data_frame,
    text="--:--:--",
    font=("Arial", 14)
)

time_value.grid(row=2, column=1, padx=20)


# -----------------------------------------------------
# COUNTERS
# -----------------------------------------------------

counter_frame = ttk.LabelFrame(
    root,
    text="Detection Counter"
)

counter_frame.pack(
    padx=30,
    pady=15,
    fill="x"
)


ttk.Label(
    counter_frame,
    text="Normal",
    font=("Arial", 12)
).grid(row=0, column=0, padx=35, pady=15)

normal_value = ttk.Label(
    counter_frame,
    text="0",
    font=("Arial", 20, "bold")
)

normal_value.grid(row=1, column=0)


ttk.Label(
    counter_frame,
    text="DoS",
    font=("Arial", 12)
).grid(row=0, column=1, padx=35, pady=15)

dos_value = ttk.Label(
    counter_frame,
    text="0",
    font=("Arial", 20, "bold")
)

dos_value.grid(row=1, column=1)


ttk.Label(
    counter_frame,
    text="Port Scan",
    font=("Arial", 12)
).grid(row=0, column=2, padx=35, pady=15)

portscan_value = ttk.Label(
    counter_frame,
    text="0",
    font=("Arial", 20, "bold")
)

portscan_value.grid(row=1, column=2)


# -----------------------------------------------------
# RESET BUTTON
# -----------------------------------------------------

reset_button = ttk.Button(
    root,
    text="Reset Counter",
    command=reset_counter
)

reset_button.pack(pady=20)


# =====================================================
# START MQTT
# =====================================================

thread = threading.Thread(
    target=mqtt_thread,
    daemon=True
)

thread.start()


# =====================================================
# START GUI
# =====================================================

root.mainloop()
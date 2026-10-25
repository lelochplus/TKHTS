from flask import Flask, jsonify, request
import paho.mqtt.client as mqtt
from datetime import datetime
import threading

# =========================================================
# CONFIG
# =========================================================

MQTT_BROKER = "192.168.1.43"
MQTT_PORT = 1883
MQTT_TOPIC = "gateway/alert"

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

# =========================================================
# DATA
# =========================================================

count_normal = 0
count_dos = 0
count_portscan = 0

current_class = -1
current_label = "No Data"
last_alert_time = "--"

data_lock = threading.Lock()


# =========================================================
# MQTT
# =========================================================

def on_connect(client, userdata, flags, rc):
    print("MQTT connected, rc =", rc)

    if rc == 0:
        client.subscribe(MQTT_TOPIC)
        print("Subscribed:", MQTT_TOPIC)
    else:
        print("MQTT connection failed!")


def on_message(client, userdata, msg):
    global count_normal
    global count_dos
    global count_portscan
    global current_class
    global current_label
    global last_alert_time

    try:
        payload = msg.payload.decode("utf-8").strip()

        print("MQTT:", payload)

        import json

        data = json.loads(payload)

        cls = int(data.get("class", -1))
        label = data.get("label", "unknown")

        with data_lock:

            current_class = cls
            current_label = label

            last_alert_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            if cls == 0:
                count_normal += 1

            elif cls == 1:
                count_dos += 1

            elif cls == 2:
                count_portscan += 1

    except Exception as e:
        print("MQTT data error:", e)


mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION1
)

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message


def mqtt_thread():
    try:
        mqtt_client.connect(
            MQTT_BROKER,
            MQTT_PORT,
            60
        )

        mqtt_client.loop_forever()

    except Exception as e:
        print("MQTT ERROR:", e)


# =========================================================
# API
# =========================================================

@app.route("/api/status")
def api_status():

    with data_lock:

        total = (
            count_normal
            + count_dos
            + count_portscan
        )

        return jsonify({
            "class": current_class,
            "label": current_label,
            "last_alert": last_alert_time,

            "normal": count_normal,
            "dos": count_dos,
            "portscan": count_portscan,
            "total": total
        })


@app.route("/api/reset", methods=["POST"])
def api_reset():

    global count_normal
    global count_dos
    global count_portscan

    with data_lock:

        count_normal = 0
        count_dos = 0
        count_portscan = 0

    print("Counters reset")

    return jsonify({
        "success": True
    })


# =========================================================
# WEB DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    return """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>RISC-V + AI Accelerator</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f6f8;

    color: #222;
}

.header {

    background: #1f2937;

    color: white;

    text-align: center;

    padding: 25px;
}

.header h1 {

    margin: 0;

    font-size: 32px;
}

.header p {

    margin: 8px 0 0;

    font-size: 20px;

    opacity: 0.85;
}

.container {

    max-width: 1100px;

    margin: 30px auto;

    padding: 0 20px;
}

.card {

    background: white;

    border-radius: 12px;

    padding: 25px;

    margin-bottom: 25px;

    box-shadow:
        0 2px 10px
        rgba(0,0,0,0.08);
}

.card-title {

    font-size: 20px;

    font-weight: bold;

    margin-bottom: 20px;
}

.status {

    text-align: center;

    padding: 25px;

    border-radius: 10px;

    font-size: 32px;

    font-weight: bold;

    background: #e5e7eb;

    color: #374151;
}

.status.normal {

    background: #dcfce7;

    color: #166534;
}

.status.dos {

    background: #fee2e2;

    color: #991b1b;
}

.status.portscan {

    background: #ffedd5;

    color: #9a3412;
}

.info {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 20px;

    margin-top: 20px;
}

.info-box {

    background: #f8fafc;

    padding: 20px;

    border-radius: 10px;

    text-align: center;
}

.info-title {

    font-size: 15px;

    color: #64748b;

    margin-bottom: 8px;
}

.info-value {

    font-size: 22px;

    font-weight: bold;
}

.counter-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 15px;
}

.counter {

    text-align: center;

    padding: 25px 10px;

    border-radius: 10px;

    background: #f8fafc;
}

.counter-title {

    font-size: 16px;

    color: #64748b;

    margin-bottom: 10px;
}

.counter-value {

    font-size: 36px;

    font-weight: bold;
}

button {

    margin-top: 20px;

    width: 100%;

    padding: 14px;

    border: none;

    border-radius: 8px;

    background: #374151;

    color: white;

    font-size: 17px;

    cursor: pointer;
}

button:hover {

    background: #111827;
}

.footer {

    text-align: center;

    color: #64748b;

    padding: 20px;
}

@media (max-width: 700px) {

    .counter-grid {

        grid-template-columns:
            repeat(2, 1fr);
    }

    .info {

        grid-template-columns:
            1fr;
    }

    .header h1 {

        font-size: 25px;
    }

}

</style>

</head>


<body>


<div class="header">

    <h1>RISC-V + AI Accelerator</h1>

    <p>IoT Gateway</p>

</div>


<div class="container">


    <!-- CURRENT STATUS -->

    <div class="card">

        <div class="card-title">
            Current Status
        </div>

        <div id="status"
             class="status">

            WAITING FOR DATA

        </div>

    </div>


    <!-- CURRENT DETECTION -->

    <div class="card">

        <div class="card-title">
            Current Detection
        </div>

        <div class="info">

            <div class="info-box">

                <div class="info-title">
                    Class
                </div>

                <div id="class"
                     class="info-value">
                    --
                </div>

            </div>


            <div class="info-box">

                <div class="info-title">
                    Last Alert
                </div>

                <div id="time"
                     class="info-value">
                    --
                </div>

            </div>

        </div>

    </div>


    <!-- COUNTERS -->

    <div class="card">

        <div class="card-title">
            Detection Counter
        </div>

        <div class="counter-grid">


            <div class="counter">

                <div class="counter-title">
                    Normal
                </div>

                <div id="normal"
                     class="counter-value">
                    0
                </div>

            </div>


            <div class="counter">

                <div class="counter-title">
                    DoS
                </div>

                <div id="dos"
                     class="counter-value">
                    0
                </div>

            </div>


            <div class="counter">

                <div class="counter-title">
                    Port Scan
                </div>

                <div id="portscan"
                     class="counter-value">
                    0
                </div>

            </div>


            <div class="counter">

                <div class="counter-title">
                    Total
                </div>

                <div id="total"
                     class="counter-value">
                    0
                </div>

            </div>

        </div>


        <button onclick="resetCounter()">

            Reset Counter

        </button>

    </div>


</div>


<div class="footer">

    RISC-V + AI Accelerator |
    IoT Gateway

</div>


<script>


function updateDashboard() {

    fetch("/api/status")

        .then(response => response.json())

        .then(data => {


            document.getElementById(
                "class"
            ).innerText =
                data.class;


            document.getElementById(
                "time"
            ).innerText =
                data.last_alert;


            document.getElementById(
                "normal"
            ).innerText =
                data.normal;


            document.getElementById(
                "dos"
            ).innerText =
                data.dos;


            document.getElementById(
                "portscan"
            ).innerText =
                data.portscan;


            document.getElementById(
                "total"
            ).innerText =
                data.total;


            const status =
                document.getElementById(
                    "status"
                );


            status.className =
                "status";


            if (data.class === 0) {

                status.innerText =
                    "NORMAL";

                status.classList.add(
                    "normal"
                );

            }

            else if (data.class === 1) {

                status.innerText =
                    "DOS ATTACK";

                status.classList.add(
                    "dos"
                );

            }

            else if (data.class === 2) {

                status.innerText =
                    "PORT SCAN";

                status.classList.add(
                    "portscan"
                );

            }

            else {

                status.innerText =
                    "WAITING FOR DATA";

            }

        })

        .catch(error => {

            console.log(
                "Dashboard error:",
                error
            );

        });

}


function resetCounter() {

    fetch(
        "/api/reset",
        {
            method: "POST"
        }
    )

    .then(() => {

        updateDashboard();

    });

}


setInterval(
    updateDashboard,
    1000
);


updateDashboard();


</script>


</body>

</html>
"""


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    thread = threading.Thread(
        target=mqtt_thread,
        daemon=True
    )

    thread.start()

    print()
    print("========================================")
    print(" RISC-V + AI Accelerator")
    print(" IoT Gateway Web Dashboard")
    print("========================================")
    print()
    print(
        "Local: "
        "http://localhost:5000"
    )
    print()
    print(
        "LAN: "
        "http://192.168.1.43:5000"
    )
    print()

    app.run(
        host=WEB_HOST,
        port=WEB_PORT,
        debug=False
    )
import paho.mqtt.client as mqtt
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os
from datetime import datetime

# Create a unique CSV filename with timestamp
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CSV_FILE = f"earthquake_data_{timestamp_str}.csv"

# Create CSV file with headers
with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'magnitude', 'vibration(10 decimal places)'])



import config_sensor
# MQTT configuration
MQTT_BROKER = config_sensor.MQTT_BROKER
MQTT_PORT = config_sensor.MQTT_PORT
MQTT_TOPIC = config_sensor.MQTT_TOPIC   

# Data storage
timestamps = []
magnitudes = []
vibrations = []

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        mag = float(data['magnitude'])
        vib = round(float(data['vibration_mm_s']), 10)
        timestamp = data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        timestamps.append(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f'))
        magnitudes.append(mag)
        vibrations.append(vib)

        # Append to CSV
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, mag, f"{vib:.10f}"])
    except Exception as e:
        print("Error:", e)

# MQTT setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# Live plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

def animate(i):
    ax1.clear()
    ax2.clear()
    if timestamps:
        ax1.plot(timestamps, magnitudes, marker='o')
        ax1.set_title("Magnitude over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Magnitude")
        ax1.grid(True)
        
        ax2.plot(timestamps, vibrations, marker='o', color='orange')
        ax2.set_title("Vibration over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Vibration")
        ax2.grid(True)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.tight_layout()
plt.show()

client.loop_stop()
client.disconnect()

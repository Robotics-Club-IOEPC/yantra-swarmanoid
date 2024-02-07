# communication.py

import paho.mqtt.client as mqtt
import math
import time
import threading

# Constants
MQTT_BROKER = "192.168.1.80"
MQTT_PORT = 1883

# Establish a global MQTT client connection
client = mqtt.Client()


def connect_mqtt():
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()


def send_mqtt_command(topic, command):
    """Publish a command to the specified MQTT topic."""
    client.publish(topic, command)


def calculate_distances(robot_corners, next_position):
    """Calculate distances from the robot's corners and center to the goal position."""
    center, tl, tr = robot_corners
    goal = next_position

    d_center = math.hypot(center[0] - goal[0], center[1] - goal[1])
    d_left = math.hypot(tl[0] - goal[0], tl[1] - goal[1])
    d_right = math.hypot(tr[0] - goal[0], tr[1] - goal[1])

    return d_right, d_left, d_center


def move_towards_goal(d_right, d_left, d_center, topic, threshold=5):
    # Threshold (t) is a tuning parameter to adjust responsiveness

    # Robot is facing towards the goal
    print(f"{d_right},{d_center},{d_left}")
    if d_center < d_right and d_center < d_left:
        send_mqtt_command(topic, "backwards")
        if d_right > d_left:
            send_mqtt_command(topic, "fast_left")
        else:
            send_mqtt_command(topic, "fast_right")

    # Robot is facing away from the goal
    else:
        if d_right > d_left and d_right - d_left > 3 * threshold:
            send_mqtt_command(topic, "fast_left")
        elif d_right > d_left and d_right - d_left > threshold:
            send_mqtt_command(topic, "left")
        elif d_left > d_right and d_left - d_right > 3 * threshold:
            send_mqtt_command(topic, "fast_right")
        elif d_left > d_right and d_left - d_right > threshold:
            send_mqtt_command(topic, "right")
        else:
            send_mqtt_command(topic, "forward")


def send_command(topic, tr, tl, robot_center, path):
    """Process the path and send appropriate MQTT commands to move the robot."""

    # Iterate through the path points and send commands
    def command_thread(topic, tr, tl, robot_center, path):
        for next_position in path:
            d_right, d_left, d_center = calculate_distances(
                (robot_center, tl, tr), next_position
            )
            move_towards_goal(float(d_right), float(d_left), float(d_center), topic)
            time.sleep(1)  # Adjust the sleep time as needed

    thread = threading.Thread(
        target=command_thread, args=(topic, tr, tl, robot_center, path)
    )
    thread.daemon = True
    thread.start()


def disconnect_mqtt():
    """Clean up the MQTT connection."""
    client.loop_stop()
    client.disconnect()

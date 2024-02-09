import cv2
import numpy as np
import threading
import paho.mqtt.client as mqtt
import math
import time
from aruco_detection import detect_aruco_markers
from astar import astar
from frameCorrection import get_warped_frame

# Define constants and setup
ARENA_WIDTH = 300
ARENA_HEIGHT = 300
GRID_SIZE = 2  # Adjust based on your setup
MQTT_BROKER = "192.168.1.97"
MQTT_PORT = 1883
CORNER_MARKERS = {0, 1, 2, 3}
INORGANIC_DROP_OFF_ID = 4
ORGANIC_DROP_OFF_ID = 5
ROBOT_IDS = [6, 7]
INORGANIC_WASTE_ID = [9, 11, 13, 15, 17]
ORGANIC_WASTE_ID = [8, 10, 12, 14, 16]

# Define PID constants and speeds for each robot
robot_settings = {
    6: {  # Robot ID 6
        "P_left": 0.005,
        "P_right": 0.005,
        "P_center": 0.001,
        "I_left": 0.1,
        "I_right": 0.1,
        "I_center": 0.1,
        "D_left": 0.01,
        "D_right": 0.01,
        "D_center": 0.01,
        "backward_speed_left": 40,  # Example speed value
        "backward_speed_right": 20,  # Example speed value
        "left_prev_error" : 0,
        "right_prev_error" : 0,
        "center_prev_error" : 0,
        "dt" : 0.1
    },
    7: {  # Robot ID 7
        "P_left": 1.5,
        "P_right": 1.5,
        "P_center": 0.6,
        "I_left": 0.3,
        "I_right": 0.3,
        "I_center": 0.07,
        "D_left": 0.02,
        "D_right": 0.02,
        "D_center": 0.02,
        "backward_speed_left": 80,  # Different speed value for robot 7
        "backward_speed_right": 60,  # Different speed value for robot 7
        "left_prev_error" : 0,
        "right_prev_error" : 0,
        "center_prev_error" : 0,
        "dt" : 0.1
    }
}


client = mqtt.Client()

# Initialize shared resources and a lock
shared_resources = {
    "frame": None,
    "markers": {},
    "drop_off_locations": {},
    "paths": {},
    "goal_positions": {},
}
resources_lock = threading.Lock()


def connect_mqtt():
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()


def send_mqtt_command(topic, command):
    client.publish(topic, command)


def calculate_distances(robot_corners, next_position):
    center, tl, tr = robot_corners
    goal = next_position
    d_center = math.hypot(center[0] - goal[0], center[1] - goal[1])
    d_left = math.hypot(tl[0] - goal[0], tl[1] - goal[1])
    d_right = math.hypot(tr[0] - goal[0], tr[1] - goal[1])
    return d_right, d_left, d_center


def move_towards_goal(robot_id, path, threshold=10):
    """
    Move the robot towards the goal following the path.
    """
    settings = robot_settings[robot_id]
    P_left = settings["P_left"]
    P_right = settings["P_right"]
    P_center = settings["P_center"]
    I_left = settings["I_left"]
    I_right = settings["I_right"]
    I_center = settings["I_center"]
    D_left = settings["D_left"]
    D_right = settings["D_right"]
    D_center = settings["D_center"]
    backward_speed_left = settings["backward_speed_left"]
    backward_speed_right = settings["backward_speed_right"]
    left_prev_error = settings["left_prev_error"]
    right_prev_error = settings["right_prev_error"]
    center_prev_error = settings["center_prev_error"]
    dt = settings["dt"]

    print(settings)

    for next_position in path:
        position_reached = False
        while not position_reached:
            with resources_lock:
                # Extracting robot head position, top left, top right corners, and robot center
                _, tl, tr, robot_center = get_head_position(
                    robot_id, shared_resources["markers"]
                )
                if tl is None or tr is None or robot_center is None:
                    time.sleep(0)
                    continue

                # Update the goal position for the current robot
                shared_resources["goal_positions"][robot_id] = next_position

            # Pass the robot center along with corners to calculate distances
            d_right, d_left, d_center = calculate_distances(
                (robot_center, tl, tr), next_position
            )
            print(
                f"robot ID: {robot_id},left: {d_left}, right: {d_right}, center: {d_center}"
            )

            # Determine movement command based on distances
            if d_center < min(d_right, d_left):
                send_mqtt_command(f"/robot{robot_id}_left_backward", backward_speed_left)
                send_mqtt_command(f"/robot{robot_id}_right_backward", backward_speed_right)
                print("rotate")
            else:
                left_error = d_left - d_right
                right_error = d_right - d_left
                d_center_error = d_center

                left_P_gain = P_left * left_error
                right_P_gain = P_right * right_error
                center_P_gain = P_center * d_center_error
                left_I_gain = I_left * (left_error + left_prev_error) * dt
                right_I_gain = I_right * (right_error + right_prev_error) * dt
                center_I_gain = P_center * (d_center_error + center_prev_error) * dt
                left_D_gain = (D_left * (left_error - left_prev_error)) / dt
                right_D_gain = (D_right * (right_error - right_prev_error)) / dt
                center_D_gain = (D_center * (d_center_error - center_prev_error)) / dt
                center_prev_error = d_center_error
                right_prev_error = right_error
                left_prev_error = left_error

                left_speed = (
                    left_P_gain
                    + left_I_gain
                    + left_D_gain
                    + center_P_gain
                    + center_I_gain
                    + center_D_gain
                )
                right_speed = (
                    right_P_gain
                    + right_I_gain
                    + right_D_gain
                    + center_P_gain
                    + center_I_gain
                    + center_D_gain
                )

                print(right_speed)
                print(left_speed)

                if left_speed >= 0:
                    send_mqtt_command(f"/robot{robot_id}_left_forward", left_speed)
                if left_speed < 0:
                    send_mqtt_command(
                        f"/robot{robot_id}_left_backward", abs(left_speed)
                    )
                if right_speed >= 0:
                    send_mqtt_command(f"/robot{robot_id}_right_forward", right_speed)
                if right_speed < 0:
                    send_mqtt_command(
                        f"/robot{robot_id}_right_backward", abs(right_speed)
                    )

            # Check if the robot has reached the next position
            with resources_lock:
                current_position, _, _, _ = get_head_position(
                    robot_id, shared_resources["markers"]
                )

                # Update the goal position for the current robot
                shared_resources["goal_positions"][robot_id] = next_position

                if (
                    current_position
                    and math.hypot(
                        current_position[0] - next_position[0],
                        current_position[1] - next_position[1],
                    )
                    < 20
                ):
                    position_reached = True

            time.sleep(0.1)  # Adjust sleep time as needed


def draw_lines_to_goal(
    frame, robot_corners, goal_position, color=(255, 0, 0), thickness=2
):
    # Unpack the robot_corners tuple
    center, tl, tr = robot_corners

    # Check if tl or tr is None and return the frame unmodified if true
    if tl is None or tr is None:
        print("Skipping drawing lines to goal due to missing corner information.")
        return frame

    # Draw lines from each corner to the goal
    cv2.line(
        frame,
        (int(tl[0]), int(tl[1])),
        (int(goal_position[0]), int(goal_position[1])),
        color,
        thickness,
    )
    cv2.line(
        frame,
        (int(tr[0]), int(tr[1])),
        (int(goal_position[0]), int(goal_position[1])),
        color,
        thickness,
    )

    # Also draw a line from the center to the goal
    cv2.line(
        frame,
        (int(center[0]), int(center[1])),
        (int(goal_position[0]), int(goal_position[1])),
        (0, 0, 255),
        thickness,
    )

    # Optionally, mark the goal position and the corners
    cv2.circle(
        frame, (int(goal_position[0]), int(goal_position[1])), 5, (0, 0, 255), -1
    )  # Goal position in red
    cv2.circle(
        frame, (int(tl[0]), int(tl[1])), 5, (255, 0, 0), -1
    )  # Top-left corner in blue
    cv2.circle(
        frame, (int(tr[0]), int(tr[1])), 5, (255, 0, 0), -1
    )  # Top-right corner in blue

    return frame


def draw_path(frame, path, color, thickness=2, grid_size=15):
    """Draws a path on the frame."""
    if not path:
        return

    # Draw each segment of the path
    for i in range(len(path) - 1):
        cv2.line(frame, path[i], path[i + 1], color, thickness)
        cv2.circle(frame, path[i], 2, (0, 0, 0), 1)


def get_head_position(robot_id, markers):
    """
    Return the head position, the top left and top right corner positions,
    and the center of the marker based on ArUco marker detection.
    """
    if robot_id in markers:
        for marker_data in markers[robot_id]:
            # Assuming corners are provided in the order: tl, tr, br, bl
            corners = marker_data["corners"]
            tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

            # Calculate the midpoint between tl and tr for the head position
            head_position = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2))

            # Calculate the center of the marker as the average of all corners
            center_x = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
            center_y = int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
            marker_center = (center_x, center_y)

            # Ensure tl and tr are tuples of integers
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))

            return head_position, tl, tr, marker_center
    return None, None, None, None


def get_waste_positions(markers, waste_id):
    """Filter and return positions of a specific waste type."""
    return [
        data["center"]
        for marker_id, marker_list in markers.items()
        if marker_id == waste_id
        for data in marker_list
    ]


def fill_grid_cells_from_corners(corners, grid_size=5):
    """Given corners of a rectangular area, returns all grid cells covered by the rectangle."""
    # Convert each corner into grid coordinates
    grid_corners = [
        convert_to_grid_coordinates(corner, cell_size=grid_size) for corner in corners
    ]

    min_x = min(corner[0] for corner in grid_corners)
    max_x = max(corner[0] for corner in grid_corners)
    min_y = min(corner[1] for corner in grid_corners)
    max_y = max(corner[1] for corner in grid_corners)

    # Fill in the grid cells
    covered_cells = set()
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            covered_cells.add((x, y))

    return covered_cells


def update_obstacles(markers, target_waste_ids, robot_head_pos):
    obstacles = set()
    nearest_waste_pos = None
    nearest_waste_dist = float("inf")
    # Initialize tl and tr with None to handle the case where they might not be set
    nearest_tl = None
    nearest_tr = None

    for marker_id, marker_data_list in markers.items():
        if marker_id in target_waste_ids:
            for marker_data in marker_data_list:
                corners = marker_data["corners"]
                tl, tr = corners[0], corners[1]
                bl, br = corners[3], corners[2]
                right_center = ((tr[0] + br[0]) / 2, (tr[1] + br[1]) / 2)
                left_center = ((tl[0] + bl[0]) / 2, (tl[1] + bl[1]) / 2)
                head_center = ((tl[0] + tr[0]) / 2, (tl[1] + tr[1]) / 2)
                obstacles.add(head_center)
                distance = np.linalg.norm(
                    np.array(robot_head_pos) - np.array(head_center)
                )
                if distance < nearest_waste_dist:
                    nearest_waste_dist = distance
                    nearest_waste_pos = head_center
                    nearest_tl = tl  # Update nearest_tl and nearest_tr with the corners of the nearest marker
                    nearest_tr = tr

    if nearest_waste_pos:
        obstacles.discard(nearest_waste_pos)

    # Return the corners of the nearest waste marker along with other return values
    return obstacles, nearest_waste_pos, nearest_tl, nearest_tr


def convert_to_grid_coordinates(position, cell_size=15):
    """Converts position to grid coordinates."""
    if not isinstance(position, tuple) or len(position) != 2:
        raise ValueError("Position must be a tuple of (x, y).")
    grid_x = int(position[0] / cell_size)
    grid_y = int(position[1] / cell_size)
    return (grid_x, grid_y)


def convert_obstacles_to_grid(obstacles, cell_size=15):
    """Converts a set of positions to grid coordinates."""
    grid_obstacles = set()
    for position in obstacles:
        if not isinstance(position, tuple) or len(position) != 2:
            raise ValueError("Each position must be a tuple of (x, y).")
        grid_x = int(position[0] / cell_size)
        grid_y = int(position[1] / cell_size)
        grid_obstacles.add((grid_x, grid_y))
    return grid_obstacles


def convert_grid_to_actual(path, cell_size=15):
    """Converts a path of grid coordinates back to actual coordinates."""
    actual_path = [
        (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
        for x, y in path
    ]
    return actual_path


def plan_path(start, goal, obstacles):
    """Wrapper for the A* pathfinding."""
    start_grid = convert_to_grid_coordinates(start)
    goal_grid = convert_to_grid_coordinates(goal)

    obstacles = convert_obstacles_to_grid(obstacles)

    # Assuming your grid size is the width/height of the arena divided by GRID_SIZE
    grid_size = ARENA_WIDTH // GRID_SIZE

    path = astar(start_grid, goal_grid, obstacles, grid_size)
    return path  # This will be a list of grid coordinates representing the path


def calculate_approach_point(marker_head_pos, marker_orientation, approach_distance=20):
    # Normalize the orientation vector
    norm = np.sqrt(marker_orientation[0] ** 2 + marker_orientation[1] ** 2)
    direction = (marker_orientation[0] / norm, marker_orientation[1] / norm)

    # Calculate the approach point
    approach_point = (
        marker_head_pos[0] + direction[0] * approach_distance,
        marker_head_pos[1] + direction[1] * approach_distance,
    )

    return approach_point


def pickup_waste(robot_id):
    pass


def drop_off_waste(robot_id):
    pass


def robot_control_loop(robot_id):
    global shared_resources, resources_lock
    # Connect to MQTT
    connect_mqtt()

    while True:
        # Acquire frame and markers
        with resources_lock:
            frame = shared_resources.get("frame", None)
            markers = shared_resources.get("markers", {})
            drop_off_locations = shared_resources.get("drop_off_locations", {})

        if frame is None:
            continue

        (
            robot_head_pos,
            robot_top_left_corner,
            robot_top_right_corner,
            _,
        ) = get_head_position(robot_id, markers)

        if not robot_head_pos:
            time.sleep(0)
            continue

        # Determine target waste and calculate path to waste
        target_waste_ids = ORGANIC_WASTE_ID if robot_id == 6 else INORGANIC_WASTE_ID
        obstacles, nearest_waste_pos, tl, tr = update_obstacles(
            markers, target_waste_ids, robot_head_pos
        )

        path_to_waste, path_to_drop_off = [], []
        if nearest_waste_pos:
            # Calculate marker orientation (simplified, adjust based on your data structure)
            marker_orientation = (tr[0] - tl[0], tr[1] - tl[1])
            approach_point = calculate_approach_point(
                nearest_waste_pos, marker_orientation
            )
            print(approach_point)
            path_to_waste = plan_path(robot_head_pos, nearest_waste_pos, obstacles)
            path_to_waste = (
                convert_grid_to_actual(path_to_waste) if path_to_waste else []
            )

            # Calculate path to drop-off only if waste is found
            drop_off_id = (
                ORGANIC_DROP_OFF_ID
                if target_waste_ids == ORGANIC_WASTE_ID
                else INORGANIC_DROP_OFF_ID
            )
            drop_off_location = drop_off_locations.get(drop_off_id)
            if drop_off_location:
                path_to_drop_off = plan_path(
                    nearest_waste_pos, drop_off_location, obstacles
                )
                path_to_drop_off = (
                    convert_grid_to_actual(path_to_drop_off) if path_to_drop_off else []
                )

        # Update shared resources with calculated paths and head position
        with resources_lock:
            shared_resources["paths"][robot_id] = {
                "path_to_waste": path_to_waste,
                "path_to_drop_off": path_to_drop_off,
            }

        if path_to_waste:
            move_towards_goal(robot_id, path_to_waste)  # Move towards waste
            send_mqtt_command(f"/robot{robot_id}_right_forward", 0)
            send_mqtt_command(f"/robot{robot_id}_left_forward", 0)
            time.sleep(5)
            pickup_waste(robot_id)  # Simulate waste pickup
        if path_to_drop_off:
            move_towards_goal(robot_id, path_to_drop_off)  # Move towards drop-off
            drop_off_waste(robot_id)  # Simulate waste drop-off

        # Loop with a delay to prevent constant recalculating
        time.sleep(0)

    # Disconnect MQTT when done
    disconnect_mqtt()


def capture_and_update_shared_resources(url):
    global shared_resources, resources_lock
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform frame correction here
        PAD = 8
        markerTL = 0
        markerTR = 2
        markerBL = 1
        markerBR = 3

        # Define the markers and their positions
        marker_ids = [markerTL, markerTR, markerBL, markerBR]
        corrected_frame, marker_corners_dict = get_warped_frame(frame, marker_ids, PAD)

        # if corrected_frame is not None:
        #     frame = corrected_frame  # Use the corrected frame for further processing

        markers = detect_aruco_markers(frame)  # Detect ArUco markers in the frame
        with resources_lock:
            shared_resources["frame"] = frame
            shared_resources["markers"] = markers
            shared_resources["drop_off_locations"] = {
                INORGANIC_DROP_OFF_ID: markers.get(INORGANIC_DROP_OFF_ID)[0]["center"]
                if markers.get(INORGANIC_DROP_OFF_ID)
                else None,
                ORGANIC_DROP_OFF_ID: markers.get(ORGANIC_DROP_OFF_ID)[0]["center"]
                if markers.get(ORGANIC_DROP_OFF_ID)
                else None,
            }


def visualize_robot_behavior():
    global shared_resources, resources_lock
    while True:
        with resources_lock:
            frame = shared_resources.get("frame", None)
            paths = shared_resources.get("paths", {})
            markers = shared_resources.get("markers", {})
            goal_positions = shared_resources.get("goal_positions", {})

            if frame is None:
                continue

            frame_copy = frame.copy()
            for robot_id in ROBOT_IDS:
                (
                    robot_head_pos,
                    robot_top_left_corner,
                    robot_top_right_corner,
                    _,
                ) = get_head_position(robot_id, markers)
                if robot_head_pos:
                    # Draw robot head position
                    cv2.circle(
                        frame_copy,
                        robot_head_pos,
                        radius=5,
                        color=(255, 0, 0),
                        thickness=-1,
                    )

            for robot_id, path_info in paths.items():
                draw_path(
                    frame_copy,
                    path_info["path_to_waste"],
                    (125, 125, 255),
                    2,
                    GRID_SIZE,
                )
                draw_path(
                    frame_copy,
                    path_info["path_to_drop_off"],
                    (125, 155, 125),
                    2,
                    GRID_SIZE,
                )

            for robot_id in ROBOT_IDS:
                (
                    robot_head_pos,
                    robot_top_left_corner,
                    robot_top_right_corner,
                    robot_center,
                ) = get_head_position(robot_id, markers)

                # Check if there is a current goal position for the robot
                if robot_id in goal_positions:
                    goal_position = goal_positions[robot_id]
                    draw_lines_to_goal(
                        frame_copy,
                        (robot_center, robot_top_left_corner, robot_top_right_corner),
                        goal_position,
                    )

            for marker_id, marker_data in markers.items():
                for data in marker_data:
                    corners = data["corners"]
                    cv2.polylines(
                        frame_copy,
                        [np.array(corners, np.int32).reshape((-1, 1, 2))],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    cv2.circle(
                        frame_copy,
                        data["center"],
                        radius=2,
                        color=(0, 0, 255),
                        thickness=-1,
                    )

            cv2.imshow("Robot Visualization", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    # Start the video capture and shared resources update in a separate thread
    capture_thread = threading.Thread(
        target=capture_and_update_shared_resources,
        # args=("http://127.0.0.1:5000/video_feed",),
        args=("http://192.168.1.68:8080/video",),
        # args=("http://10.155.42.112:8080/video",),
        # args=(0,),
        daemon=True,
    )
    capture_thread.start()

    # Start a thread for each robot
    robot_threads = [
        threading.Thread(target=robot_control_loop, args=(robot_id,), daemon=True)
        for robot_id in ROBOT_IDS
    ]
    for thread in robot_threads:
        thread.start()

    # Visualization thread
    visualization_thread = threading.Thread(
        target=visualize_robot_behavior, daemon=True
    )
    visualization_thread.start()

    # Wait for the capture thread to finish
    capture_thread.join()

    # Threads are daemon threads, so they will exit when the main thread exits
    # Ensure all windows are closed properly
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

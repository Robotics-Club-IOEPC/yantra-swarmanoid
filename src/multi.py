import cv2
import numpy as np
import threading
from aruco_detection import detect_aruco_markers
from astar import astar
from communication import connect_mqtt, send_command, disconnect_mqtt
import time

# Define constants and setup
ARENA_WIDTH = 300
ARENA_HEIGHT = 300
GRID_SIZE = 2  # Adjust based on your setup

# Marker IDs
CORNER_MARKERS = {0, 1, 2, 3}
INORGANIC_DROP_OFF_ID = 4
ORGANIC_DROP_OFF_ID = 5
ROBOT_IDS = [6, 7]
INORGANIC_WASTE_ID = [8, 9, 10, 11, 12]
ORGANIC_WASTE_ID = [13, 14, 15, 16, 17]

# Initialize shared resources and a lock
shared_resources = {"frame": None, "markers": {}, "drop_off_locations": {}}
resources_lock = threading.Lock()


def draw_path(frame, path, color, thickness=2, grid_size=15):
    """Draws a path on the frame."""
    if not path:
        return

    # Draw each segment of the path
    for i in range(len(path) - 1):
        cv2.line(frame, path[i], path[i + 1], color, thickness)


def get_head_position(robot_id, markers):
    """Return the head position and the top left and top right corner positions of the marker based on ArUco marker detection."""
    if robot_id in markers:
        for marker_data in markers[robot_id]:
            # Assuming corners are provided in the order: tl, tr, br, bl
            corners = marker_data["corners"]
            tl, tr = corners[0], corners[1]

            # Calculate the midpoint between tl and tr for the head position
            head_position = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2))
            # Ensure tl and tr are tuples of integers
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))

            return head_position, tl, tr
    return None, None, None


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
    """Update and return the obstacles and the target waste position based on detected markers."""
    obstacles = set()
    nearest_waste_pos = None
    nearest_waste_dist = float("inf")

    # Add all waste markers as obstacles
    for marker_id, marker_data_list in markers.items():
        if marker_id in INORGANIC_WASTE_ID or marker_id in ORGANIC_WASTE_ID:
            for marker_data in marker_data_list:
                center = marker_data["center"]
                obstacles.add(center)

    # Find the nearest target waste marker
    for marker_data_list in markers.get(target_waste_ids[0], []):
        center = marker_data_list["center"]
        distance = np.linalg.norm(np.array(robot_head_pos) - np.array(center))
        if distance < nearest_waste_dist:
            nearest_waste_dist = distance
            nearest_waste_pos = center

    # Remove the nearest target waste position from obstacles if it is present
    if nearest_waste_pos:
        obstacles.discard(nearest_waste_pos)

    return obstacles, nearest_waste_pos


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


def pickup_waste(robot_id):
    pass


def drop_off_waste(robot_id):
    pass


def robot_control_loop(robot_id):
    global shared_resources, resources_lock
    # Connect to MQTT
    connect_mqtt()
    while True:
        with resources_lock:
            frame = shared_resources.get("frame", None)
            markers = shared_resources.get("markers", {})
            drop_off_locations = shared_resources.get("drop_off_locations", {})

        if frame is None:
            continue

        frame_copy = frame.copy()

        (
            robot_head_pos,
            robot_top_left_corner,
            robot_top_right_corner,
        ) = get_head_position(robot_id, markers)
        if robot_head_pos:
            cv2.circle(
                frame_copy, robot_head_pos, radius=5, color=(255, 0, 0), thickness=-1
            )

            target_waste_ids = ORGANIC_WASTE_ID if robot_id == 6 else INORGANIC_WASTE_ID
            obstacles, nearest_waste_pos = update_obstacles(
                markers, target_waste_ids, robot_head_pos
            )

            for obstacle in obstacles:
                obstacle_text = f" {obstacle}"
                cv2.putText(
                    frame_copy,
                    obstacle_text,
                    obstacle,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            if nearest_waste_pos:
                path_to_waste = plan_path(robot_head_pos, nearest_waste_pos, obstacles)
                if path_to_waste:
                    # Correctly convert path to actual pixel coordinates for drawing
                    path_to_waste = convert_grid_to_actual(path_to_waste)
                    print(
                        f"Robot Head Center: {robot_head_pos}, Robot Top Left Corner: {robot_top_left_corner}, Robot Top Right Corner: {robot_top_right_corner}"
                    )
                    print(f"Bot Path: {path_to_waste}")
                    draw_path(frame_copy, path_to_waste, (125, 125, 255), 2, GRID_SIZE)
                    # to send_command with the topic based on robot_id
                    topic = f"/bot{robot_id - 5}"  # Assuming robot_id 6 -> 'bot1', robot_id 7 -> 'bot2'
                    send_command(
                        topic,
                        robot_top_right_corner,
                        robot_top_left_corner,
                        robot_head_pos,
                        path_to_waste,
                    )
                    pickup_waste(robot_id)

                drop_off_id = (
                    ORGANIC_DROP_OFF_ID
                    if target_waste_ids == ORGANIC_WASTE_ID
                    else INORGANIC_DROP_OFF_ID
                )
                drop_off_location = drop_off_locations[drop_off_id]

                if drop_off_location:
                    path_to_drop_off = plan_path(
                        nearest_waste_pos, drop_off_location, obstacles
                    )
                    if path_to_drop_off:
                        path_to_drop_off = convert_grid_to_actual(path_to_drop_off)
                        draw_path(
                            frame_copy,
                            path_to_drop_off,
                            (125, 155, 125),
                            2,
                            GRID_SIZE,
                        )
                        # to send_command with the topic based on robot_id
                        topic = f"/bot{robot_id - 5}"  # Assuming robot_id 6 -> 'bot1', robot_id 7 -> 'bot2'
                        send_command(
                            topic,
                            robot_top_right_corner,
                            robot_top_left_corner,
                            robot_head_pos,
                            path_to_drop_off,
                        )
                        drop_off_waste(robot_id)

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

            cv2.imshow(f"Robot {robot_id} Control View", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(0.1)
    # Disconnect from MQTT when done
    disconnect_mqtt()


def capture_and_update_shared_resources(url):
    global shared_resources, resources_lock
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
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


def main():
    # Start the video capture and shared resources update in a separate thread
    capture_thread = threading.Thread(
        target=capture_and_update_shared_resources,
        args=("http://127.0.0.1:5000/video_feed",),
        # args=("http://192.168.1.68:4747/video",),
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

    # Wait for the capture thread to finish
    capture_thread.join()

    # Threads are daemon threads, so they will exit when the main thread exits
    # Ensure all windows are closed properly
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

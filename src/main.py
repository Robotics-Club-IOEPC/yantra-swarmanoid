import cv2
import numpy as np
from aruco_detection import detect_aruco_markers
from astar import astar
from communication import send_command

# Define constants and setup
ARENA_WIDTH = 300
ARENA_HEIGHT = 300
GRID_SIZE = 2  # Adjust based on your setup
# Marker IDs
CORNER_MARKERS = {0, 1, 2, 3}
INORGANIC_DROP_OFF_ID = 4
ORGANIC_DROP_OFF_ID = 5
ROBOT_IDS = [6, 7]
# ROBOT_IDS = [7]
INORGANIC_WASTE_ID = [8, 9, 10, 11, 12]
ORGANIC_WASTE_ID = [13, 14, 15, 16, 17]


def draw_path(frame, path, color, thickness=2, grid_size=15):
    """Draws a path on the frame."""
    if not path:
        return
    # Convert path from grid coordinates back to pixel coordinates
    pixel_path = [
        (x * grid_size + grid_size // 2, y * grid_size + grid_size // 2)
        for x, y in path
    ]
    # Draw each segment of the path
    for i in range(len(pixel_path) - 1):
        cv2.line(frame, pixel_path[i], pixel_path[i + 1], color, thickness)


def get_head_position(robot_id, markers):
    """Return the head position of the marker based on ArUco marker detection."""
    if robot_id in markers:
        for marker_data in markers[robot_id]:
            # Assuming corners are provided in the order: tl, tr, br, bl
            tl, tr = marker_data["corners"][0], marker_data["corners"][1]

            # Calculate the midpoint between tl and tr for the head position
            head_position = (int(tl[0] + tr[0]) // 2, int(tl[1] + tr[1]) // 2)

            return head_position
    return None


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


url = "http://127.0.0.1:5000/video_feed"


# url = 0
def main():
    cap = cv2.VideoCapture(url)  # Adjust the source based on your setup
    paths = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        markers = detect_aruco_markers(frame)  # Detect ArUco markers in the frame

        # Iterate over each detected marker and draw a box around it
        for marker_id, marker_data in markers.items():
            for data in marker_data:
                corners = data["corners"]
                # Since corners are already a list of tuples, we can use them directly.
                # We need to make the corners list cyclic to connect the last point to the first.
                cv2.polylines(
                    frame,
                    [np.array(corners, np.int32).reshape((-1, 1, 2))],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

                # Optionally, draw the center point
                cv2.circle(
                    frame, data["center"], radius=2, color=(0, 0, 255), thickness=-1
                )

        drop_off_locations = {
            INORGANIC_DROP_OFF_ID: markers.get(INORGANIC_DROP_OFF_ID)[0]["center"]
            if markers.get(INORGANIC_DROP_OFF_ID)
            else None,
            ORGANIC_DROP_OFF_ID: markers.get(ORGANIC_DROP_OFF_ID)[0]["center"]
            if markers.get(ORGANIC_DROP_OFF_ID)
            else None,
        }

        # Calculate paths for each robot
        for robot_id in ROBOT_IDS:
            robot_head_pos = get_head_position(robot_id, markers)
            if not robot_head_pos:
                continue

            # Draw the robot's head position as a blue circle
            cv2.circle(frame, robot_head_pos, radius=5, color=(255, 0, 0), thickness=-1)

            # Decide which IDs to look for based on the robot ID
            target_waste_ids = ORGANIC_WASTE_ID if robot_id == 6 else INORGANIC_WASTE_ID

            # Update obstacles and get the nearest waste position
            obstacles, nearest_waste_pos = update_obstacles(
                markers, target_waste_ids, robot_head_pos
            )

            # Now use nearest_waste_pos as the target for pathfinding and obstacles for obstacle avoidance
            if nearest_waste_pos:
                # Exclude the nearest waste and other robots as obstacles
                robot_obstacles = obstacles
                # Draw obstacle positions on the screen
                for obstacle in robot_obstacles:
                    obstacle_text = f" {obstacle}"
                    cv2.putText(
                        frame,
                        obstacle_text,
                        obstacle,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                path_to_waste = plan_path(
                    robot_head_pos, nearest_waste_pos, robot_obstacles
                )

                if path_to_waste:
                    # Store the path to prevent other robots from taking it
                    paths[robot_id] = set(path_to_waste)
                    draw_path(frame, path_to_waste, (0, 0, 255))  # Blue path to waste
                    send_command(robot_id, path_to_waste)

                    # After reaching the waste (this logic needs to be implemented based on your robot's feedback mechanism)
                    pickup_waste(robot_id)

                    # Calculate path to the drop-off location considering other robot paths
                    drop_off_id = (
                        ORGANIC_DROP_OFF_ID
                        if target_waste_ids == ORGANIC_WASTE_ID
                        else INORGANIC_DROP_OFF_ID
                    )

                    path_to_drop_off = plan_path(
                        nearest_waste_pos,
                        drop_off_locations[drop_off_id],
                        robot_obstacles,
                    )

                    if path_to_drop_off:
                        draw_path(
                            frame, path_to_drop_off, (0, 255, 0)
                        )  # Green path to drop off
                        send_command(robot_id, path_to_drop_off)
                        drop_off_waste(robot_id)

        # Show the frame with detected markers for debugging
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

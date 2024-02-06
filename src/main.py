import cv2
import numpy as np
from aruco_detection import detect_aruco_markers
from astar import astar
from communication import send_command

# Define constants and setup
ARENA_WIDTH = 300
ARENA_HEIGHT = 300
GRID_SIZE = 5  # Adjust based on your setup

# Marker IDs
CORNER_MARKERS = {0, 1, 2, 3}
INORGANIC_DROP_OFF_ID = 4
ORGANIC_DROP_OFF_ID = 5
ROBOT_IDS = [6, 7]
INORGANIC_WASTE_ID = 8
ORGANIC_WASTE_ID = 9


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


def draw_obstacle(frame, obstacle, color, thickness=1, grid_size=15):
    """Draws a rectangle for the obstacle grid cell on the frame."""
    top_left = (obstacle[0] * grid_size, obstacle[1] * grid_size)
    bottom_right = ((obstacle[0] + 1) * grid_size, (obstacle[1] + 1) * grid_size)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)


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


def find_nearest_waste(robot_head_pos, waste_positions):
    """Find and return the nearest waste position from the robot."""
    nearest_position = None
    min_distance = float("inf")
    for pos in waste_positions:
        distance = np.linalg.norm(np.array(robot_head_pos) - np.array(pos))
        if distance < min_distance:
            nearest_position = pos
            min_distance = distance
    return nearest_position


def fill_grid_cells_from_corners(corners, grid_size=15):
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


def update_obstacles(markers, exclude_ids):
    """Update and return the list/set of obstacles based on detected markers, excluding specific IDs."""
    all_exclude_ids = set(exclude_ids).union(
        CORNER_MARKERS, {INORGANIC_DROP_OFF_ID, ORGANIC_DROP_OFF_ID}
    )
    obstacles = set()
    for marker_id, marker_data_list in markers.items():
        if marker_id not in all_exclude_ids:
            for marker_data in marker_data_list:
                corners = marker_data["corners"]
                obstacles.update(
                    fill_grid_cells_from_corners(corners, grid_size=GRID_SIZE)
                )
    return obstacles


def convert_to_grid_coordinates(position, cell_size=15):
    """Converts position to grid coordinates."""
    if not isinstance(position, tuple) or len(position) != 2:
        raise ValueError("Position must be a tuple of (x, y).")
    grid_x = int(position[0] / cell_size)
    grid_y = int(position[1] / cell_size)
    return (grid_x, grid_y)


def plan_path(start, goal, obstacles):
    """Wrapper for the A* pathfinding."""
    start_grid = convert_to_grid_coordinates(start)
    goal_grid = convert_to_grid_coordinates(goal)

    # Convert obstacle positions to grid coordinates
    obstacle_grid = {convert_to_grid_coordinates(obstacle) for obstacle in obstacles}

    # Assuming your grid size is the width/height of the arena divided by GRID_SIZE
    grid_size = ARENA_WIDTH // GRID_SIZE

    path = astar(start_grid, goal_grid, obstacle_grid, grid_size)
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

        # Show the frame with detected markers for debugging
        # Initialize obstacles for all robots
        common_obstacles = update_obstacles(markers, exclude_ids=[])

        # Draw obstacles
        # for obstacle in common_obstacles:
        #     draw_obstacle(
        #         frame, obstacle, (128, 0, 128), grid_size=GRID_SIZE
        #     )  # Purple color for obstacles

        # Calculate paths for each robot
        for robot_id in ROBOT_IDS:
            robot_head_pos = get_head_position(robot_id, markers)
            if not robot_head_pos:
                continue

            # Draw the robot's head position as a blue circle
            cv2.circle(frame, robot_head_pos, radius=5, color=(255, 0, 0), thickness=-1)

            target_waste_id = ORGANIC_WASTE_ID if robot_id == 6 else INORGANIC_WASTE_ID
            waste_positions = get_waste_positions(markers, target_waste_id)
            nearest_waste_pos = find_nearest_waste(robot_head_pos, waste_positions)

            if nearest_waste_pos:
                # Exclude the nearest waste and other robots as obstacles
                robot_obstacles = common_obstacles.union(paths.get(robot_id, set()))

                # # Redraw obstacles to update the view after excluding the nearest waste
                for obstacle in robot_obstacles:
                    draw_obstacle(
                        frame, obstacle, (128, 0, 128), grid_size=GRID_SIZE
                    )  # Purple color for obstacles

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
                        if target_waste_id == ORGANIC_WASTE_ID
                        else INORGANIC_DROP_OFF_ID
                    )
                    robot_obstacles = common_obstacles.union(paths.get(robot_id, set()))

                    # # Redraw obstacles to update the view after excluding the nearest waste
                    # for obstacle in robot_obstacles:
                    #     draw_obstacle(
                    #         frame, obstacle, (128, 0, 128), grid_size=GRID_SIZE
                    #     )  # Purple color for obstacles

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

import heapq
import math


# Manhattan Distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Euclidean Distance
# def heuristic(a, b):
#     return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# Diagonal Distance
# def heuristic(a, b):
#     dx = abs(a[0] - b[0])
#     dy = abs(a[1] - b[1])
#     return max(dx, dy)


# Chebyshev Distance
# def heuristic(a, b):
#     dx = abs(a[0] - b[0])
#     dy = abs(a[1] - b[1])
#     return max(dx, dy, dx + dy)


def astar(start, goal, obstacles, grid_size):
    # Define the neighbors function
    def neighbors(position, obstacles, grid_size, obstacle_radius):
        x, y = position
        candidates = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y - 1),
            (x + 1, y + 1),
        ]

        valid_neighbors = []
        for neighbor in candidates:
            nx, ny = neighbor
            if (
                0 <= nx < grid_size
                and 0 <= ny < grid_size
                and neighbor not in obstacles
                and all(
                    abs(ox - nx) > obstacle_radius or abs(oy - ny) > obstacle_radius
                    for ox, oy in obstacles
                )
            ):
                valid_neighbors.append(neighbor)
        return valid_neighbors

    # Initialize the data structures
    open_set = []
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # Add the start position to the open set
    heapq.heappush(open_set, (f_score[start], start))

    while open_set:
        # Get the position with the lowest f-score from the open set
        current = heapq.heappop(open_set)[1]

        if current == goal:
            # Reconstruct the path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        # Add the current position to the closed set
        closed_set.add(current)

        # Explore the neighbors
        for neighbor in neighbors(current, obstacles, grid_size, obstacle_radius=2):
            neighbor_g_score = g_score[current] + 1

            if neighbor in closed_set:
                continue

            if neighbor not in open_set or neighbor_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = neighbor_g_score
                f_score[neighbor] = neighbor_g_score + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    return None

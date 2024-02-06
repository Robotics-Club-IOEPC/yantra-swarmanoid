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


def get_neighbors(node, obstacles, grid_size):
    """Generate neighbors for a given node, excluding obstacles."""
    directions = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (-1, -1),
        (-1, +1),
        (+1, +1),
        (+1, -1),
    ]  # 4-way connectivity
    result = []
    for d in directions:
        neighbor = (node[0] + d[0], node[1] + d[1])
        # Check if the neighbor is within bounds and not an obstacle
        if (
            0 <= neighbor[0] < grid_size
            and 0 <= neighbor[1] < grid_size
            and neighbor not in obstacles
        ):
            result.append(neighbor)
    return result


def astar(start, goal, obstacles, grid_size):
    """Implement the A* algorithm."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in get_neighbors(current, obstacles, grid_size):
            tentative_g_score = (
                g_score[current] + 1
            )  # Assume cost between neighbors is 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # Path not found

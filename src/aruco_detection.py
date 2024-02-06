import cv2
import numpy as np


def calculate_scale(corners, marker_physical_size_cm=15):
    """
    Calculate the scale in pixels per cm based on the distance between the top left and top right corners.

    :param corners: The corners of the marker.
    :param marker_physical_size_cm: The physical size of the marker in centimeters.
    :return: Scale in pixels per cm.
    """
    # Calculate the distance between the top left and top right corners in pixels
    tl, tr = corners[0], corners[1]
    pixel_distance = np.linalg.norm(np.array(tl) - np.array(tr))

    # Calculate the scale as pixels per cm
    scale = pixel_distance / marker_physical_size_cm
    return scale


def adjust_marker_corners(
    corners,
    offset_x_cm=0,
    offset_y_cm=0,
    adjust_width_cm=0,
    adjust_height_cm=0,
    marker_physical_size_cm=15,
):
    """
    Adjust the corners of a marker by offsets and resizing based on centimeters.

    :param corners: Original corners of the marker.
    :param offset_x_cm: Horizontal offset in cm.
    :param offset_y_cm: Vertical offset in cm.
    :param adjust_width_cm: Adjustment to width in cm.
    :param adjust_height_cm: Adjustment to height in cm.
    :param marker_physical_size_cm: The physical size of the marker in centimeters for scale calculation.
    :return: Adjusted corners.
    """
    scale = calculate_scale(corners, marker_physical_size_cm)

    # Convert cm adjustments to pixels
    offset_x_pixels = offset_x_cm * scale
    offset_y_pixels = offset_y_cm * scale
    adjust_width_pixels = adjust_width_cm * scale
    adjust_height_pixels = adjust_height_cm * scale

    # Assuming top-left, top-right, bottom-right, bottom-left order
    tl, tr, br, bl = corners

    # Center of the marker
    center_x, center_y = np.mean(corners, axis=0)

    # Adjust for width and height
    tl = (tl[0] - adjust_width_pixels // 2, tl[1] - adjust_height_pixels // 2)
    tr = (tr[0] + adjust_width_pixels // 2, tr[1] - adjust_height_pixels // 2)
    br = (br[0] + adjust_width_pixels // 2, br[1] + adjust_height_pixels // 2)
    bl = (bl[0] - adjust_width_pixels // 2, bl[1] + adjust_height_pixels // 2)

    # Apply offset
    adjusted_corners = [
        (x + offset_x_pixels, y + offset_y_pixels) for x, y in [tl, tr, br, bl]
    ]

    return adjusted_corners


def detect_aruco_markers(frame, aruco_dict_type=cv2.aruco.DICT_6X6_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()  # Updated for some versions of OpenCV
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        frame, aruco_dict, parameters=parameters
    )

    markers = {}
    if ids is not None:
        ids = ids.flatten()
        for id, corner in zip(ids, corners):
            # Process corners to a more readable format
            processed_corners = [
                tuple(map(int, corner_point)) for corner_point in corner[0]
            ]

            # Apply adjustments for specific markers
            if id == 4 or id == 5 or id == 6 or id == 7:
                processed_corners = adjust_marker_corners(
                    processed_corners,
                    offset_x_cm=33.75 if id == 5 else 0,
                    offset_y_cm=33.75 if id == 4 else 0,
                    adjust_width_cm=60 if id == 4 else (33.75 if id == 5 else 15),
                    adjust_height_cm=33.75 if id == 4 else (60 if id == 5 else 15),
                )

            # Recalculate the center based on the processed corners
            recalculated_center = tuple(map(int, np.mean(processed_corners, axis=0)))

            # Store both center and corners
            marker_data = {"center": recalculated_center, "corners": processed_corners}

            if id in markers:
                markers[id].append(
                    marker_data
                )  # Append the new marker data to the list for this ID
            else:
                markers[id] = [marker_data]  # Start a new list with this marker data

    return markers

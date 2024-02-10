import cv2
import numpy as np


def calculate_scale(corners, marker_physical_size_cm):
    # Calculate the distance between the first and second corner (top-left and top-right) in pixels
    pixel_distance = np.linalg.norm(np.array(corners[0]) - np.array(corners[1]))
    # Scale is pixels per cm
    return pixel_distance / marker_physical_size_cm


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
    # Calculate the center of the marker
    center = np.mean(corners, axis=0)

    # Calculate the vectors from the center to the corners
    vectors = corners - center

    # Calculate the scale based on the physical size of the marker
    scale = np.linalg.norm(vectors[0]) / (marker_physical_size_cm / 2)

    # Convert cm adjustments to pixels
    offset_x_pixels = offset_x_cm * scale
    offset_y_pixels = offset_y_cm * scale
    adjust_width_pixels = adjust_width_cm * scale
    adjust_height_pixels = adjust_height_cm * scale

    # Adjust the width in the marker's local coordinate system
    width_adjustment_factor = 1 + adjust_width_cm / marker_physical_size_cm
    vectors[:, 0] *= width_adjustment_factor

    # Adjust the height in the marker's local coordinate system
    height_adjustment_factor = 1 + adjust_height_cm / marker_physical_size_cm
    vectors[:, 1] *= height_adjustment_factor

    # Rotate the offset to the marker's coordinate system if needed
    # This is optional and depends on whether you want the offset to rotate with the marker
    # If not, you can simply add the offset to the center point
    rotated_offset = center + np.array([offset_x_pixels, offset_y_pixels])

    # Apply the adjustments and offsets to get the new corners
    adjusted_corners = vectors + rotated_offset

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
                    offset_x_cm=22.5 if id == 5 else 0,
                    offset_y_cm=22.5 if id == 4 else 0,
                    adjust_width_cm=0 if id == 4 else (0 if id == 5 else 15),
                    adjust_height_cm=0 if id == 4 else (0 if id == 5 else 15),
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

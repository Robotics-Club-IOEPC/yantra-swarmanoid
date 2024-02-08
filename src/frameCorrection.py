import cv2
import numpy as np
import cv2.aruco as aruco

# Define the dictionary to use
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Create ArUco parameters
parameters = cv2.aruco.DetectorParameters()

def get_warped_frame(input_frame, marker_ids, PAD):
    # Detect markers in the frame
    corners, ids, _ = cv2.aruco.detectMarkers(
        input_frame, aruco_dict, parameters=parameters
    )

    marker_corners_dict = {marker_id: None for marker_id in marker_ids}

    if ids is not None:
        # Extract corners of the specified markers
        for marker_id in marker_ids:
            index = np.where(ids == marker_id)
            if len(index[0]) > 0:
                marker_corners_dict[marker_id] = corners[index[0][0]][0]

    # Check if all specified markers are detected
    if all(value is not None for value in marker_corners_dict.values()):
        # Get the width and height of the frame
        frame_width, frame_height = input_frame.shape[1], input_frame.shape[0]

        # Define the corners of the big marker
        big_marker_corners = [
            marker_corners_dict[marker_ids[3]][2],  # TL = 0
            marker_corners_dict[marker_ids[1]][1],  # BL = 3
            marker_corners_dict[marker_ids[0]][0],  # BR = 2
            marker_corners_dict[marker_ids[2]][3],  # TR = 1
        ]

        square_size = min(frame_width, frame_height)
        destination_points = np.float32(
            [
                [square_size - PAD, square_size - PAD],
                [square_size - PAD, 0 + PAD],
                [0 + PAD, 0 + PAD],
                [0 + PAD, square_size - PAD],
            ]
        )

        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(
            np.float32(big_marker_corners), destination_points
        )

        # Warp the frame using the perspective transformation matrix
        warped_frame = cv2.warpPerspective(
            input_frame, matrix, (square_size, square_size)
        )

        return warped_frame, marker_corners_dict

    return None, None  # Return None if not all markers are detected

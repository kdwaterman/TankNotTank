# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import cv2
import numpy as np
from tflite_support.task import processor
# Constants for colors
GREEN = (0, 255, 0)  # Green color for bounding boxes, center dot, and connecting line
RED = (0, 0, 255)  # Red color for center dot in the highest probability detection box
LINE_THICKNESS = 1  # Thickness for the dashed line
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1





# Modified visualize function
def visualize(image: np.ndarray, detection_result) -> np.ndarray:
    # Draw bounding boxes and labels on the image
    highest_probability = 0
    highest_prob_box = None

    for detection in detection_result.detections:
        # Extract bounding box coordinates
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        # Draw bounding box in green
        cv2.rectangle(image, start_point, end_point, GREEN, 2)
        
        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, GREEN, FONT_THICKNESS)

        # Identify the detection box with the highest probability
        if probability > highest_probability:
            highest_probability = probability
            highest_prob_box = (bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, 			bbox.origin_y + bbox.height)
            


    # Draw a red dot at the center of the highest probability detection box
    if highest_prob_box:
        center_x = (highest_prob_box[0] + highest_prob_box[2]) // 2
        center_y = (highest_prob_box[1] + highest_prob_box[3]) // 2
        cv2.circle(image, (center_x, center_y), 5, RED, -1)

        # Draw a green dot at the center of the camera's field of view
        height, width = image.shape[:2]
        center_of_view = (width // 2, height // 2)
        cv2.circle(image, center_of_view, 5, GREEN, -1)

        # Draw a thin dashed green line connecting the two dots
        draw_dashed_line(image, center_of_view, (center_x, center_y), GREEN, LINE_THICKNESS)

    return image


def draw_dashed_line(img, start, end, color, thickness=1, dash_length=5):
    # Function to draw a dashed line on the image
    dX = end[0] - start[0]
    dY = end[1] - start[1]
    dashes = []
    for i in range(0, 1000, dash_length * 2):
        phase = float(i) / 1000
        x = int(start[0] + (dX * phase))
        y = int(start[1] + (dY * phase))
        dashes.append((x, y))
    for i in range(len(dashes) - 1):
        cv2.line(img, dashes[i], dashes[i + 1], color, thickness)

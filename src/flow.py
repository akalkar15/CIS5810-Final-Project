import cv2
import numpy as np

def classify_horizontal_vertical(flow_x_avg, flow_y_avg):
    if abs(flow_x_avg) > abs(flow_y_avg):
        if flow_x_avg > 0:
            return "Right"
        else:
            return "Left"
    else:
        if flow_y_avg > 0:
            return "Down"
        else:
            return "Up"

def get_direction(video_path):
    print("Tracking direction of object... ")
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        exit()
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    direction_counts = {"Up": 0, "Down": 0, "Left": 0, "Right": 0}
    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **farneback_params)
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        flow_x_avg = np.mean(flow_x)
        flow_y_avg = np.mean(flow_y)

        direction = classify_horizontal_vertical(flow_x_avg, flow_y_avg)
        direction_counts[direction] += 1
        #print(f"General Direction: {direction}")
        prev_gray = gray

    most_frequent_direction = max(direction_counts, key=direction_counts.get)
    return most_frequent_direction


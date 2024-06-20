import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import tempfile
from sort import Sort

# Load YOLO model
model = YOLO("yolov8l.pt")

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define lane limits
limits = {
    'limit1': [935, 90, 1275, 90],
    'limit2': [935, 110, 1275, 110],
    'limit3': [1365, 120, 1365, 360],
    'limit4': [1385, 120, 1385, 360],
    'limit5': [600, 70, 600, 170],
    'limit6': [620, 70, 620, 170],
    'limit7': [450, 500, 1240, 500],
    'limit8': [450, 520, 1240, 520],
}

# Initialize counts
totalCounts = {1: [], 2: [], 3: [], 4: []}

# Default frame skip value and step
frame_skip = 5
frame_skip_step = 1

# Variable to track current frame index
current_frame_index = 0

def process_frame(img):
    imgRegion = cv2.bitwise_and(img, img)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for key, coords in limits.items():
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), (250, 182, 122), 2)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(111, 237, 235))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(15, y1)), scale=0.6, thickness=1, colorR=(56, 245, 213), colorT=(25, 26, 25), offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

        for i in range(1, 5):
            if (limits[f'limit{2*i-1}'][0] < cx < limits[f'limit{2*i-1}'][2] and
                limits[f'limit{2*i-1}'][1] - 15 < cy < limits[f'limit{2*i-1}'][1] + 15) or \
               (limits[f'limit{2*i}'][0] < cx < limits[f'limit{2*i}'][2] and
                limits[f'limit{2*i}'][1] - 15 < cy < limits[f'limit{2*i}'][1] + 15):
                if totalCounts[i].count(id) == 0:
                    totalCounts[i].append(id)
                    cv2.line(img, (limits[f'limit{2*i-1}'][0], limits[f'limit{2*i-1}'][1]),
                             (limits[f'limit{2*i-1}'][2], limits[f'limit{2*i-1}'][3]), (12, 202, 245), 3)
                    cv2.line(img, (limits[f'limit{2*i}'][0], limits[f'limit{2*i}'][1]),
                             (limits[f'limit{2*i}'][2], limits[f'limit{2*i}'][3]), (12, 202, 245), 3)

    for i in range(1, 5):
        cvzone.putTextRect(img, f'{i}st Lane: {len(totalCounts[i])}', (25, 50 + 60 * (i-1)), 0.6, thickness=1, colorR=(147, 245, 186), colorT=(15, 15, 15))

    return img

st.title("Traffic Flow Monitoring")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    frame_skipped = 0

    while cap.isOpened():
        if frame_skipped < frame_skip:
            frame_skipped += 1
            cap.grab()
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_skipped = 0
        frame = process_frame(frame)
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

# UI for adjusting frame skip
st.sidebar.subheader("Adjust Frame Skipping")
if st.sidebar.button("Increase Frame Skip"):
    frame_skip += frame_skip_step
    st.sidebar.write(f"Current Frame Skip: {frame_skip}")
if st.sidebar.button("Decrease Frame Skip"):
    frame_skip = max(1, frame_skip - frame_skip_step)
    st.sidebar.write(f"Current Frame Skip: {frame_skip}")

from ultralytics import YOLO
import cv2
import time
import os

# Configuration
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.50
SAVE_FOLDER = "../outputs/saved_frames"

# Setup
os.makedirs(SAVE_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# FPS variables
prev_time = time.time()

# Main loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Run YOLO inference
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # Copy frame for drawing
    display_frame = frame.copy()

    # Draw detections manually
    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().numpy()
        )

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        # Bounding box
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            display_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # FPS calculation
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # Display FPS
    cv2.putText(
        display_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Instructions
    cv2.putText(
        display_frame,
        "Press Q to quit | S to save frame",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show window
    cv2.imshow(
        "Real-Time YOLO Detection",
        display_frame
    )

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Save frame
    elif key == ord("s"):
        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = os.path.join(
            SAVE_FOLDER,
            f"frame_{timestamp}.jpg"
        )

        cv2.imwrite(
            filename,
            display_frame
        )

        print(
            f"Frame saved: {filename}"
        )

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Webcam released successfully.")
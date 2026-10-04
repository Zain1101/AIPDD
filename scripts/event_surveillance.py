from ultralytics import YOLO
import cv2
import time
import os
import csv

# Configuration
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.65

# COCO class IDs:
# 0  = person
# 67 = cell phone
HIGH_PRIORITY_CLASSES = {
    0: "person",
    67: "cell phone"
}

ALERT_FOLDER = "../alerts"
LOG_FOLDER = "../logs"
LOG_FILE = os.path.join(LOG_FOLDER, "events.log")


# Create required folders
os.makedirs(ALERT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)


# Initialize log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "Class Name",
            "Confidence",
            "Bounding Box"
        ])

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


print("Surveillance system started.")
print("Press Q to quit.")


# Cooldown control
last_alert_time = 0

ALERT_COOLDOWN = 5
# prevents saving hundreds of images
# for the same object every frame

# Main loop
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break


    # Run YOLO
    results = model(
        frame,
        verbose=False
    )

    result = results[0]

    display_frame = frame.copy()

    alert_triggered = False
    alert_details = []

    # Process detections
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Check if high-priority object
        if (
            class_id in HIGH_PRIORITY_CLASSES
            and confidence > CONFIDENCE_THRESHOLD
        ):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].cpu().numpy()
            )

            class_name = HIGH_PRIORITY_CLASSES[class_id]

            alert_triggered = True

            alert_details.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })


            # Draw RED bounding box
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            # Detection label
            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                display_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # Trigger logging
    current_time = time.time()

    if (
        alert_triggered
        and current_time - last_alert_time >= ALERT_COOLDOWN
    ):

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        readable_timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Save snapshot
        snapshot_filename = os.path.join(
            ALERT_FOLDER,
            f"{timestamp}.jpg"
        )

        cv2.imwrite(
            snapshot_filename,
            display_frame
        )

        # Append to log file
        with open(
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            for event in alert_details:

                x1, y1, x2, y2 = event["bbox"]

                writer.writerow([
                    readable_timestamp,
                    event["class_name"],
                    f"{event['confidence']:.3f}",
                    f"{x1},{y1},{x2},{y2}"
                ])


        print(
            f"ALERT logged: {snapshot_filename}"
        )

        last_alert_time = current_time

    # Visual indicator
    if alert_triggered:

        cv2.rectangle(
            display_frame,
            (0, 0),
            (display_frame.shape[1], 60),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display_frame,
            "SECURITY EVENT DETECTED",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    else:

        cv2.rectangle(
            display_frame,
            (0, 0),
            (display_frame.shape[1], 60),
            (0, 150, 0),
            -1
        )

        cv2.putText(
            display_frame,
            "MONITORING - NO EVENT",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # Display live feed
    cv2.imshow(
        "Edge-Triggered Surveillance",
        display_frame
    )
    # Quit control
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
cap.release()

cv2.destroyAllWindows()

print("Surveillance system stopped.")
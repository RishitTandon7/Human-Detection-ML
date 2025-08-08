import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt or yolov8m.pt for better accuracy

# Open webcam (0 = default cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]

    person_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]

        # Class 0 = Person in COCO dataset
        if cls == 0:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw red dot (pointing system) at center top (like head)
            cv2.circle(frame, (cx, y1 + 10), 5, (0, 0, 255), -1)

            # Optional: Add ID or label
            cv2.putText(frame, f'Human {person_count}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display total count
    cv2.putText(frame, f'Total Humans: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Human Tracking with Red Dots", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


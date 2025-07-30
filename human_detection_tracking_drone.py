import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import random

def main():
    # Initialize Hugging Face object detection pipeline
    detector = pipeline("object-detection", model="facebook/detr-resnet-50")

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)

    # Create MultiTracker object compatible with OpenCV version
    def create_multitracker():
        if hasattr(cv2, 'MultiTracker_create'):
            return cv2.MultiTracker_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'MultiTracker_create'):
            return cv2.legacy.MultiTracker_create()
        else:
            return None

    trackers = create_multitracker()
    if trackers is None:
        # Fallback: manage multiple trackers manually
        trackers = []
    tracking = False

    # Create tracker function compatible with OpenCV version
    def create_tracker():
        try:
            return cv2.TrackerCSRT_create()
        except AttributeError:
            pass
        try:
            return cv2.TrackerMIL_create()
        except AttributeError:
            pass
        try:
            return cv2.TrackerGOTURN_create()
        except AttributeError:
            pass
        try:
            return cv2.TrackerNano_create()
        except AttributeError:
            pass
        try:
            return cv2.TrackerVit_create()
        except AttributeError:
            pass
        raise AttributeError("Your OpenCV version does not support any of the available trackers")

    # Dictionary to store motion trails for each tracked ID
    motion_trails = {}

    # Generate distinct colors for each ID
    def get_color(idx):
        random.seed(idx)
        return tuple(random.randint(0, 255) for _ in range(3))

    frame_count = 0
    detection_interval = 24  # Run detection every 24 frames

    def iou(boxA, boxB):
        # Compute Intersection over Union between two boxes
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height, width = frame.shape[:2]

        frame_count += 1
        run_detection = False
        if not tracking or frame_count % detection_interval == 0:
            run_detection = True

        if run_detection:
            # Convert OpenCV BGR frame to PIL Image RGB
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run Hugging Face object detection
            results = detector(pil_image)

            new_boxes = []
            new_confidences = []

            for result in results:
                if result['label'] == 'person' and result['score'] > 0.8:
                    box = result['box']
                    x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
                    w = x2 - x1
                    h = y2 - y1
                    if w >= 50 and h >= 50:
                        new_boxes.append((x1, y1, w, h))
                        new_confidences.append(result['score'])

            # Compare new detections with existing trackers using IoU to avoid duplicates
            existing_boxes = []
            if hasattr(trackers, 'getObjects'):
                existing_boxes = [tuple(map(int, box)) for box in trackers.getObjects()]
            else:
                existing_boxes = last_boxes if 'last_boxes' in locals() else []

            # Remove trackers for people who left the frame (not updated recently)
            # For simplicity, we will rely on tracker update success below

            # Add new trackers for new detections with low IoU to existing boxes
            for i, new_box in enumerate(new_boxes):
                max_iou = 0
                for ex_box in existing_boxes:
                    iou_val = iou(new_box, ex_box)
                    if iou_val > max_iou:
                        max_iou = iou_val
                if max_iou < 0.5 and new_confidences[i] > 0.7:
                    print(f"Adding new tracker for detection {i+1} with confidence {new_confidences[i]*100:.2f}%")
                    tracker = create_tracker()
                    if hasattr(trackers, 'add'):
                        trackers.add(tracker, frame, new_box)
                    else:
                        tracker.init(frame, new_box)
                        trackers.append(tracker)
                    motion_trails[len(trackers)-1] = []

            tracking = True

        if hasattr(trackers, 'update'):
            success, boxes = trackers.update(frame)
            if success:
                last_boxes = boxes
                to_remove = []
                for i, box in enumerate(boxes):
                    x, y, w, h = [int(v) for v in box]
                    # Smooth bounding box coordinates using moving average
                    if i not in motion_trails:
                        motion_trails[i] = []
                    motion_trails[i].append((x, y, w, h))
                    if len(motion_trails[i]) > 5:
                        motion_trails[i].pop(0)
                    avg_box = np.mean(motion_trails[i], axis=0).astype(int)
                    x, y, w, h = avg_box.tolist()
                    print(f"Tracking human {i+1} at [{x}, {y}, {w}, {h}]")
                    # Update motion trail center points
                    center = (x + w // 2, y + h // 2)
                    if 'centers' not in motion_trails:
                        motion_trails['centers'] = {}
                    if i not in motion_trails['centers']:
                        motion_trails['centers'][i] = []
                    motion_trails['centers'][i].append(center)
                    if len(motion_trails['centers'][i]) > 20:
                        motion_trails['centers'][i].pop(0)
                    # Draw motion trail
                    for j in range(1, len(motion_trails['centers'][i])):
                        alpha = int(255 * (j / len(motion_trails['centers'][i])))
                        color = get_color(i)
                        overlay = frame.copy()
                        cv2.line(overlay, motion_trails['centers'][i][j - 1], motion_trails['centers'][i][j], color, 2, lineType=cv2.LINE_AA)
                        cv2.addWeighted(overlay, alpha / 255.0, frame, 1 - alpha / 255.0, 0, frame)
                    # Draw bounding box with thicker line and anti-aliased
                    color = (255, 0, 0)  # Blue color in BGR
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)
                    # Put label with confidence and "human" text
                    label = f"human {new_confidences[i]:.2f}" if 'new_confidences' in locals() and i < len(new_confidences) else "human"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                    text_w, text_h = text_size
                    # Solid background for label
                    cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w, y), color, -1)
                    # Put white text
                    cv2.putText(frame, label, (x, y - 3), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
            else:
                print("Lost tracking. Searching for humans again.")
                trackers = create_multitracker() if hasattr(cv2, 'MultiTracker_create') else []
                tracking = False
                motion_trails.clear()
        else:
            success_all = True
            to_remove = []
            for i, tracker in enumerate(trackers):
                success = tracker.update(frame)[0]
                box = tracker.update(frame)[1] if success else None
                if success and box is not None:
                    x, y, w, h = [int(v) for v in box]
                    print(f"Tracking human {i+1} at [{x}, {y}, {w}, {h}]")
                    color = get_color(i)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                else:
                    print(f"Lost tracking human {i+1}")
                    to_remove.append(i)
                    success_all = False
            # Remove trackers for lost humans
            for idx in reversed(to_remove):
                if hasattr(trackers, 'remove'):
                    trackers.remove(idx)
                else:
                    trackers.pop(idx)
                motion_trails.pop(idx, None)
            if not success_all:
                print("Lost tracking. Searching for humans again.")
                if not hasattr(trackers, 'remove'):
                    # Reset trackers list if any lost
                    trackers = []
                tracking = False
                motion_trails.clear()

        cv2.imshow("Human Detection and Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


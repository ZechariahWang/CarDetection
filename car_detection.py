import cv2
import numpy as np

def merge_nearby_detections(detections, max_distance=50):
    if not detections:
        return []

    detections = list(detections)
    merged = []
    used = set()

    for i, (x1, y1, w1, h1) in enumerate(detections):
        if i in used:
            continue

        group = [(x1, y1, w1, h1)]
        used.add(i)

        for j, (x2, y2, w2, h2) in enumerate(detections):
            if j <= i or j in used:
                continue

            cx1, cy1 = x1 + w1/2, y1 + h1/2
            cx2, cy2 = x2 + w2/2, y2 + h2/2
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            if distance <= max_distance:
                group.append((x2, y2, w2, h2))
                used.add(j)

        x_min = min(x for x, y, w, h in group)
        y_min = min(y for x, y, w, h in group)
        x_max = max(x + w for x, y, w, h in group)
        y_max = max(y + h for x, y, w, h in group)
        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged

def get_centroid(x, y, w, h):
    return (x + w//2, y + h//2)

def match_detections(prev_detections, curr_detections, max_distance=100):
    if not prev_detections:
        return [(i, curr) for i, curr in enumerate(curr_detections)]

    matches = []
    used = set()

    for prev_id, prev_box in prev_detections:
        px, py, pw, ph = prev_box
        prev_centroid = get_centroid(px, py, pw, ph)

        best_match = None
        best_distance = max_distance

        for j, curr_box in enumerate(curr_detections):
            if j in used:
                continue

            cx, cy, cw, ch = curr_box
            curr_centroid = get_centroid(cx, cy, cw, ch)

            distance = np.sqrt((prev_centroid[0] - curr_centroid[0])**2 +
                             (prev_centroid[1] - curr_centroid[1])**2)

            if distance < best_distance:
                best_distance = distance
                best_match = j

        if best_match is not None:
            matches.append((prev_id, curr_detections[best_match]))
            used.add(best_match)

    next_id = max([m[0] for m in matches], default=-1) + 1
    for j, curr_box in enumerate(curr_detections):
        if j not in used:
            matches.append((next_id, curr_box))
            next_id += 1

    return matches

cap = cv2.VideoCapture("other_car.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=100)

tracked_boxes = []
prev_tracked_boxes = []
fps = cap.get(cv2.CAP_PROP_FPS)

# Road-based calibration (US standard lane width: 3.7 meters)
# the road conversion algorithms were from chat gpt, idk how accruate they are 
LANE_WIDTH_METERS = 3.7  
LANE_WIDTH_PIXELS = 150  
PIXELS_PER_METER = LANE_WIDTH_PIXELS / LANE_WIDTH_METERS
METERS_PER_SECOND_TO_KMH = 3.6 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    roi=frame[300:800,10:1000]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area>1000:
            x,y,w,h = cv2.boundingRect(contour)
            detections.append((x,y,w,h))

    merged_detections = merge_nearby_detections(detections, max_distance=80)
    tracked_boxes = match_detections(tracked_boxes, merged_detections, max_distance=100)

    for box_id, (x, y, w, h) in tracked_boxes:
        cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(roi, f"ID: {box_id}", (x, y-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        prev_box = None
        for prev_id, prev_det in prev_tracked_boxes:
            if prev_id == box_id:
                prev_box = prev_det
                break

        if prev_box:
            px, py, pw, ph = prev_box
            distance_pixels = np.sqrt((x - px)**2 + (y - py)**2)
            distance_meters = distance_pixels / PIXELS_PER_METER
            speed_ms = distance_meters * fps  # meters per second
            speed_kmh = speed_ms * METERS_PER_SECOND_TO_KMH
            cv2.putText(roi, f"Speed: {speed_kmh:.1f} km/h", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    prev_tracked_boxes = tracked_boxes.copy()

    cv2.imshow("ROI", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)
    if key==27: # s
        break

cap.release()
cv2.destroyAllWindows()


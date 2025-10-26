import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_skin = np.array([0, 10, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

lower_skin2 = np.array([166, 10, 60], dtype=np.uint8)
upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)

kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def count_fingers(contour):
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 5:
            return 0

        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None or len(defects) == 0:
            return 0

        # Count defects (valleys between fingers) that are deep enough
        finger_count = sum(1 for defect in defects if defect[0][3] > 3000)

        # Add 1 because we count valleys, not fingers
        return min(finger_count + 1, 5)

    except Exception as e:
        print(f"Error in count_fingers: {e}")
        return 0

def main():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        roi_x1, roi_y1 = 50, 50
        roi_x2, roi_y2 = 400, 400
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.dilate(mask, kernel_small, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        finger_count = 0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 1500:
                finger_count = count_fingers(largest_contour)
                cv2.drawContours(roi, [largest_contour], 0, (0, 255, 0), 2)

                hull = cv2.convexHull(largest_contour)
                cv2.drawContours(roi, [hull], 0, (0, 0, 255), 2)

                cv2.putText(roi, f"Area: {int(area)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)

        cv2.putText(frame, f"Fingers: {finger_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 3)

        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(frame, "Place hand in blue rectangle", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

        cv2.imshow("Finger Detection", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

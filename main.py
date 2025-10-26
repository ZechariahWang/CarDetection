import cv2
import sys

def main():
    """
    OpenCV starter script
    """
    print("OpenCV version:", cv2.__version__)

    # Example 1: Load an image from file
    # Uncomment and modify the path if you have an image file
    # image = cv2.imread('path/to/image.jpg')
    # if image is None:
    #     print("Could not load image")
    #     return
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Example 2: Capture from webcam
    print("Attempting to open webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame")
            break

        # Display the frame
        cv2.imshow('Webcam Feed', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")

if __name__ == "__main__":
    main()

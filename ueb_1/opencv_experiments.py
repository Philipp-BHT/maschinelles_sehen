import numpy as np
import cv2


def switch_mode(image: np.ndarray, mode: int) -> np.ndarray:
    modes = [
        "No Filter",
        "Color Space: HSV",
        "Color Space: LAB",
        "Color Space: YUV",
        "Histogram Equalization",
        "Thresholding: Gaussian-Thresholding",
        "Thresholding: Otsu-Thresholding",
        "Canny edge detection"
    ]
    print("Current Mode:", modes[mode])

    if mode == 0:  # No filter
        return image
    elif mode == 1:  # Convert BGR to HSV
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif mode == 2:  # Convert BGR to LAB
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif mode == 3:  # Convert BGR to YUV
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif mode == 4:  # Histogram Equalization (convert to grayscale first)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        # Stack the original and equalized images side-by-side for comparison.
        return np.hstack((gray, equ))
    elif mode == 5:  # Gaussian-Thresholding
        # For thresholding, it's common to use a grayscale image.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return th1
    elif mode == 6:  # Otsu-Thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th2
    elif mode == 7:  # Canny Edge Detection
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    else:
        raise ValueError("Unknown filter mode")


# Open the camera stream.
cap = cv2.VideoCapture(0)
mode = 0  # start in mode 0 (no filter)
cv2.namedWindow("Filtered Image", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Check key press without blocking the loop too long.
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('0'):
        mode = 0  # No filter
    elif ch == ord('1'):
        mode = 1  # HSV
    elif ch == ord('2'):
        mode = 2  # LAB
    elif ch == ord('3'):
        mode = 3  # YUV
    elif ch == ord('4'):
        mode = 4  # Histogram Equalization
    elif ch == ord('5'):
        mode = 5  # Gaussian-Thresholding
    elif ch == ord('6'):
        mode = 6  # Otsu-Thresholding
    elif ch == ord('7'):
        mode = 7  # Canny Edge Detection
    elif ch == ord('q'):
        break

    # Process the frame using the chosen filter.
    processed = switch_mode(frame, mode)

    # Display the processed image.
    cv2.imshow("Filtered Image", processed)

# Release the capture and destroy windows.
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2

# the parameter is the device number, could also be 0 oder 2
# depending on your setup
cap = cv2.VideoCapture(0)
sift = cv2.SIFT_create()
blur_flag = False


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """
    print(np.array(gray).shape)
    print(gray)
    input()
    """
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
    if ch == ord('b'):
        print("BLUR")
        blur_flag = not blur_flag

    if blur_flag:
        kernel = np.ones((7,7),np.float32)/49
        gray = cv2.filter2D(gray,-1,kernel)

    # Display the resulting frame
    cv2.imshow('frame',gray)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

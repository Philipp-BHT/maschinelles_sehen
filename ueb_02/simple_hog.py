import numpy as np
import cv2
import math
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints):

    # convert color to gray image and extract feature in gray
    gray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)


    # compute x and y gradients (sobel kernel size 5)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)


    # compute magnitude and angle of the gradients
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        # print kp.pt, kp.size
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        #(hist, bins) = np.histogram(...)

        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size // 2)

        # Define patch window around the keypoint
        x0, x1 = x - size, x + size + 1
        y0, y1 = y - size, y + size + 1

        # Clip to image bounds
        x0, x1 = max(x0, 0), min(x1, gray.shape[1])
        y0, y1 = max(y0, 0), min(y1, gray.shape[0])

        mag_patch = magnitude[y0:y1, x0:x1]
        angle_patch = angle[y0:y1, x0:x1]

        # Use only pixels with non-zero magnitude
        valid_mask = mag_patch > 0
        angles_valid = angle_patch[valid_mask]
        mag_valid = mag_patch[valid_mask]

        hist, bins = np.histogram(
            angles_valid,
            bins=8,
            range=(0, 2 * np.pi),
            weights=mag_valid
        )

        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= np.sum(hist)

        plot_histogram(hist, bins)

        descr[count] = hist

    return descr

# in our example we only have a single keypoint
keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
test = cv2.imread('./images/hog_test/circle.jpg')
descriptor = compute_simple_hog(test, keypoints)


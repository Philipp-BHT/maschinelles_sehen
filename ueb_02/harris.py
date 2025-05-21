import cv2
import numpy as np


# Load image and convert to gray and floating point
img = cv2.imread('./images/graffiti.png')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

# Define sobel filter and use cv2.filter2D to filter the grayscale image

# YOUR CODE HERE
grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

# Compute I_xx, I_yy, I_xy and sum over all I_xx etc. 3x3 neighbors to compute
# entries of the matrix M = \sum_{3x3} [ I_xx Ixy; Ixy Iyy ]
# Note1: this results again in 3 images sumIxx, sumIyy, sumIxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently

# YOUR CODE HERE

I_xx = grad_x * grad_x
I_yy = grad_y * grad_y
I_xy = grad_x * grad_y

kernel = np.ones((3, 3), dtype=np.float32)
sumGxx = cv2.filter2D(I_xx, -1, kernel)
sumGyy = cv2.filter2D(I_yy, -1, kernel)
sumGxy = cv2.filter2D(I_xy, -1, kernel)


# Define parameter
k = 0.04
threshold = 0.01

# Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. With det(M) and trace(M)
# you can compute the resulting image containing the harris corner responses with
# harris = ...

# YOUR CODE HERE

det_M = sumGxx * sumGyy - sumGxy * sumGxy
trace_M = sumGxx + sumGyy
harris = det_M - k * (trace_M ** 2)

# Filter the harris 'image' with 'harris > threshold*harris.max()'
# this will give you the indices where values are above the threshold.
# These are the corner pixel you want to use

# YOUR CODE HERE

harris_thres = np.zeros_like(harris)
harris_thres[harris > threshold * harris.max()] = 255

# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray,3,3,k)

# intialize in black - set pixels with corners in with
harris_cv_thres = np.zeros(harris_cv.shape)
harris_cv_thres[harris_cv>threshold*harris_cv.max()]=[255]

# just for debugging to create such an image as seen
# in the assignment figure.
# img[harris>threshold*harris.max()]=[255,0,0]


# please leave this - adjust variable name if desired
# the computed difference should be 0.0
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")


# cv2.imwrite("Harris_own.png", harris_thres)
# cv2.imwrite("Harris_cv.png", harris_cv_thres)
# cv2.imwrite("Image_with_Harris.png", img)

cv2.namedWindow('Interactive Systems: Harris Corner')
while True:
    ch = cv2.waitKey(0)
    if ch == 27:
        break
    cv2.imshow('harris',harris_thres)
    cv2.imshow('harris_cv',harris_cv_thres)

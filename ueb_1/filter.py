import numpy as np
import cv2


def make_gaussian(size, fwhm = 3, center=None) -> np.ndarray:
    """ Make a square gaussian kernel.

    param size is the length of a side of the square
    param fwhm is full-width-half-maximum, which
    param center can be thought of as an effective radius.
    return np.array
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)


def convolution_2d(img, kernel) -> np.ndarray:
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # TODO write convolution of arbitrary sized convolution here
    # Attention: convolution should work with kernels of any size.
    # Tip: use Numpy as good as possible, otherwise the algorithm will take too long.
    # Tip: Use Padding, the output image should be the same size of the input without
    #      black pixels at the border  
    # I.e. do not iterate over the kernel, only over the image. The rest goes with Numpy.


    offset = int(kernel.shape[0]/2)
    newimg = np.zeros(img.shape)

    # YOUR CODE HERE

    m, n = kernel.shape
    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("Kernel dimensions should be odd")

    padded_img = np.pad(img, ((offset, offset), (offset, offset)), mode='reflect')

    rank = np.linalg.matrix_rank(kernel)

    if rank == 1:
        print("Solving using separable convolution")
        U, S, V = np.linalg.svd(kernel)
        kernel_h = U[:, 0] * np.sqrt(S[0])
        kernel_v = V[0] * np.sqrt(S[0])
        print(f"Kernel_h: {kernel_h}")
        print(f"Kernel_v: {kernel_v}")

        image = img.astype(np.float32)
        offset = int(kernel_h.shape[0] / 2)

        horizontal_result = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                horizontal_result[i, j] = np.sum(padded_img[i, j:j + m] * kernel_h)

        padded2 = np.pad(horizontal_result, ((offset, offset), (0, 0)), mode='reflect')
        newimg = np.zeros_like(horizontal_result, dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                newimg[i, j] = np.sum(padded2[i:i + m, j] * kernel_v)

    else:
        print("Solving using regular convolution")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                patch = padded_img[i:i + m, j:j + n]
                newimg[i, j] = np.sum(patch * kernel)

    return np.uint8(np.clip(np.abs(newimg), 0, 255))


if __name__ == "__main__":

    # 1. load image in grayscale
    img = cv2.imread('graffiti.png', cv2.IMREAD_GRAYSCALE)

    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    # 2 use image kernels
    gauss = convolution_2d(img, gk)
    sobel_x = convolution_2d(img, sobelmask_x)
    sobel_y = convolution_2d(img, sobelmask_y)

    # 3. compute magnitude of gradients
    mog = np.sqrt(sobel_x.astype(np.float32) ** 2 + sobel_y.astype(np.float32) ** 2)
    mog = np.uint8(np.clip(mog, 0, 255))

    # Show resulting images
    # Note: sobel_x etc. must be uint8 images to get displayed correctly astype(np.uint8)

    cv2.imshow("gauss", gauss)
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("mog", mog)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


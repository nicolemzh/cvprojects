import numpy as np
import cv2

# PART 1.1
'''
First, let's recap what a convolution is. Implement it with four for loops, then two for loops. Implement padding, with zero fill values; convolution without padding will receive partial credit. Compare it with a built-in convolution function scipy.signal.convolve2d! Then, take a picture of yourself (and read it as grayscale), write out a 9x9 box filter, and convolve the picture with the box filter. Do it with the finite difference operators Dx and Dy as well. Include the code snippets in the website!
'''
def convolution_4_loops(img, kernel):
    img_h, img_w = img.shape
    ker_h, ker_w = kernel.shape
    pad_h, pad_w = ker_h // 2, ker_w // 2 # amount of padding to add

    # pad the image with zeros on all sides and copy rest of image
    padded = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w))
    padded[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = img

    output = np.zeros((img_h, img_w))

    for i in range(img_h):
        for j in range(img_w):
            for k in range(ker_h):
                for l in range(ker_w):
                    output[i, j] += padded[i + k, j + l] * kernel[k, l]

    return output

def convolution_2_loops(img, kernel):
    img_h, img_w = img.shape
    ker_h, ker_w = kernel.shape
    pad_h, pad_w = ker_h // 2, ker_w // 2 # amount of padding to add

    # pad the image with zeros on all sides and copy rest of image
    padded = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w))
    padded[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = img

    output = np.zeros((img_h, img_w))

    for i in range(img_h):
        for j in range(img_w):
            output[i, j] = np.sum(padded[i:i + ker_h, j:j + ker_w] * kernel)

    return output

# PART 2.1
def unsharp_mask_filter(alpha, sigma=1.5):
    ksize = 9

    gaussian_1d = cv2.getGaussianKernel(ksize=9, sigma=sigma)
    gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)

    center = ksize // 2
    delta = np.zeros((ksize, ksize))
    delta[center, center] = 1.0

    unsharp_filter = (1 + alpha) * delta - alpha * gaussian_kernel

    return unsharp_filter, gaussian_kernel
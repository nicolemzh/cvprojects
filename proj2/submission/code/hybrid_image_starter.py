import matplotlib.pyplot as plt
from align_image_code import align_images, high_pass_filter, low_pass_filter

import numpy as np
from skimage import color
import cv2
from scipy.signal import convolve2d
import skimage.io as skio
import skimage.util as skutil

def save_image(image, filename):
    if image.dtype in [np.float32, np.float64]:
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        im_out_uint8 = skutil.img_as_ubyte(image_normalized)
    else:
        im_out_uint8 = skutil.img_as_ubyte(image)
    
    skio.imsave(filename, im_out_uint8)

def get_gaussian(size=9, sigma=10):
    gaussian_1d = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    return gaussian_1d @ gaussian_1d.T

def low_pass_filter(img, sigma=10):
    kernel_size = int(4 * sigma + 1)
    gaussian_kernel = get_gaussian(10, sigma=sigma)

    return convolve2d(img, gaussian_kernel, mode='same')

def high_pass_filter(img, sigma=10):
    low_pass = low_pass_filter(img, sigma)
    return img - low_pass

def fft_mag(img, title):
    fft_img = np.fft.fft2(img)
    fft_img_shifted = np.fft.fftshift(fft_img)

    log_mag = np.log(np.abs(fft_img_shifted) + 1)

    return log_mag

def create_hybrid_image(im1, im2, sigma1, sigma2, a1=5.0, a2=0.5, fft=True, output_path='output/part2_2/'):
    # grayscale
    im1 = color.rgb2gray(im1)
    im2 = color.rgb2gray(im2)

    low_freq = low_pass_filter(im1, sigma1)
    high_freq = high_pass_filter(im2, sigma2)
    
    hybrid = low_freq * a1 + high_freq * a2
    hybrid = np.clip(hybrid, 0, 1)

    # display results
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 5, 1)
    plt.imshow(im1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(2, 5, 2)
    plt.imshow(im2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    plt.subplot(2, 5, 3)
    plt.imshow(low_freq, cmap='gray')
    plt.title(f'Low-pass Filtered\n(σ={sigma1})')
    plt.axis('off')
    
    plt.subplot(2, 5, 4)
    plt.imshow(high_freq, cmap='gray')
    plt.title(f'High-pass Filtered\n(σ={sigma2})')
    plt.axis('off')
    
    plt.subplot(2, 5, 5)
    plt.imshow(hybrid, cmap='gray')
    plt.title('Hybrid Image')
    plt.axis('off')

    if fft:
        plt.subplot(2, 5, 6)
        plt.imshow(fft_mag(im1, "FFT"), cmap='gray')
        plt.title('FFT Image 1')
        plt.axis('off')

        plt.subplot(2, 5, 7)
        plt.imshow(fft_mag(im2, "FFT"), cmap='gray')
        plt.title('FFT Image 2')
        plt.axis('off')

        plt.subplot(2, 5, 8)
        plt.imshow(fft_mag(low_freq, "FFT"), cmap='gray')
        plt.title('FFT Low-pass')
        plt.axis('off')

        plt.subplot(2, 5, 9)
        plt.imshow(fft_mag(high_freq, "FFT"), cmap='gray')
        plt.title('FFT High-pass')
        plt.axis('off')

        plt.subplot(2, 5, 10)
        plt.imshow(fft_mag(hybrid, "FFT"), cmap='gray')
        plt.title('FFT Hybrid')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    save_image(hybrid, f'{output_path}/ladybug_pepperoni_hybrid.png')

    return np.clip(hybrid, 0, 1)

# First load images

# high sf
im1 = plt.imread('hybrid_python/ladybug.jpg')/255.

# low sf
im2 = plt.imread('hybrid_python/pepperoni_pizza.jpg')/255

# Next align images (this code is provided, but may be improved)
im2_aligned, im1_aligned = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 10 # low
sigma2 = 7 # high
hybrid = create_hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, a1=1.0, a2=1.5)
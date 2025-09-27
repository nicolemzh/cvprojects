import utils
from PIL import Image
import numpy as np
import skimage.io as skio
import skimage.util as skutil
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2

def save_image(image, filename):
    if image.dtype in [np.float32, np.float64]:
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        im_out_uint8 = skutil.img_as_ubyte(image_normalized)
    else:
        im_out_uint8 = skutil.img_as_ubyte(image)
    
    skio.imsave(filename, im_out_uint8)

# PART 1.1
def part_1_1(path, output_path='output/part1_1/'):
    portrait = skio.imread(path, as_gray=True)
    skio.imshow(portrait)
    skio.show()

    box_filter = np.ones((9, 9)) / 81

    # results
    box_result = utils.convolution_4_loops(portrait, box_filter)
    print('Finished box filter with 4 loops')
    box_simp_result = utils.convolution_2_loops(portrait, box_filter)
    print('Finished box filter with 2 loops')
    box_scipy = convolve2d(portrait, box_filter, mode='same', boundary='fill')
    print('Finished box filter with scipy')

    dx_result = utils.convolution_2_loops(portrait, Dx)
    print('Finished Dx filter with 2 loops')
    dy_result = utils.convolution_2_loops(portrait, Dy)
    print('Finished Dy filter with 2 loops')

    # display results
    skio.imshow(box_result, cmap='gray')
    skio.show()
    skio.imshow(box_simp_result, cmap='gray')
    skio.show()
    skio.imshow(box_scipy, cmap='gray')
    skio.show()
    skio.imshow(dx_result, cmap='gray')
    skio.show()
    skio.imshow(dy_result, cmap='gray')
    skio.show()

    save_image(box_result, f'{output_path}/box_portrait.jpg')
    save_image(box_simp_result, f'{output_path}/box_simple_portrait.jpg')
    save_image(box_scipy, f'{output_path}/box_scipy_portrait.jpg')
    save_image(dx_result, f'{output_path}/dx_portrait.jpg')
    save_image(dy_result, f'{output_path}/dy_portrait.jpg')

# PART 1.2
def part_1_2(path, output_path='output/part1_2/'):
    cameraman = skio.imread(path, as_gray=True)
    dx_cameraman_result = convolve2d(cameraman, Dx, mode='same', boundary='fill')
    dy_cameraman_result = convolve2d(cameraman, Dy, mode='same', boundary='fill')

    gradient_magnitude = np.sqrt(dx_cameraman_result**2 + dy_cameraman_result**2)
    threshold = 0.22 * np.max(gradient_magnitude)

    print(f'Threshold value: {threshold}, Maximum gradient magnitude: {np.max(gradient_magnitude)}, Maximum magnitude before: {np.max(cameraman)}')

    cameraman_edges = gradient_magnitude >= threshold

    # display results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cameraman, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(cameraman_edges, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.imshow(dx_cameraman_result, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(dy_cameraman_result, cmap='gray')

    plt.show()

    save_image(cameraman_edges, f'{output_path}/cameraman_edges.jpg')
    save_image(dx_cameraman_result, f'{output_path}/dx_cameraman.jpg')
    save_image(dy_cameraman_result, f'{output_path}/dy_cameraman.jpg')

# PART 1.3
def part_1_3(path, output_path='output/part1_3/'):
    cameraman = skio.imread(path, as_gray=True)

    gaussian_1d = cv2.getGaussianKernel(ksize=9, sigma=1.5)
    gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)

    # blur then finite differentiate
    blurred_result = convolve2d(cameraman, gaussian_kernel, mode='same', boundary='fill')
    dx_blurred = convolve2d(blurred_result, Dx, mode='same', boundary='fill')
    dy_blurred = convolve2d(blurred_result, Dy, mode='same', boundary='fill')

    gradient_magnitude_blur = np.sqrt(dx_blurred**2 + dy_blurred**2)
    threshold = 0.22 * np.max(gradient_magnitude_blur)
    edge_img = gradient_magnitude_blur >= threshold

    # single convolution
    dog_x = convolve2d(gaussian_kernel, Dx, mode='same', boundary='fill')
    dog_y = convolve2d(gaussian_kernel, Dy, mode='same', boundary='fill')
    dx_dog = convolve2d(cameraman, dog_x, mode='same', boundary='fill')
    dy_dog = convolve2d(cameraman, dog_y, mode='same', boundary='fill')

    gradient_magnitude_dog = np.sqrt(dx_dog**2 + dy_dog**2)
    threshold = 0.22 * np.max(gradient_magnitude_dog)
    edge_img_dog = gradient_magnitude_dog >= threshold

    # display results
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 6, 1)
    plt.imshow(cameraman, cmap='gray')
    plt.subplot(3, 6, 7)
    plt.imshow(blurred_result, cmap='gray')
    plt.subplot(3, 6, 8)
    plt.imshow(dx_blurred, cmap='gray')
    plt.subplot(3, 6, 9)
    plt.imshow(dy_blurred, cmap='gray')
    plt.subplot(3, 6, 10)
    plt.imshow(edge_img, cmap='gray')
    plt.subplot(3, 6, 13)
    plt.imshow(gaussian_kernel, cmap='gray')
    plt.subplot(3, 6, 14)
    plt.imshow(dog_x, cmap='gray')
    plt.subplot(3, 6, 15)
    plt.imshow(dog_y, cmap='gray')
    plt.subplot(3, 6, 16)
    plt.imshow(dx_dog, cmap='gray')
    plt.subplot(3, 6, 17)
    plt.imshow(dy_dog, cmap='gray')
    plt.subplot(3, 6, 18)
    plt.imshow(edge_img_dog, cmap='gray')

    plt.savefig(f'{output_path}/cameraman_dog_process.jpg', bbox_inches='tight')
    plt.show()

    save_image(blurred_result, f'{output_path}/cameraman_blurred.jpg')
    save_image(dx_blurred, f'{output_path}/dx_cameraman.jpg')
    save_image(dy_blurred, f'{output_path}/dy_cameraman.jpg')
    save_image(edge_img, f'{output_path}/cameraman_edges.jpg')

    save_image(gaussian_kernel, f'{output_path}/gaussian_kernel.jpg')
    save_image(dog_x, f'{output_path}/dog_x.jpg')
    save_image(dog_y, f'{output_path}/dog_y.jpg')
    save_image(dx_dog, f'{output_path}/dx_cameraman_dog.jpg')
    save_image(dy_dog, f'{output_path}/dy_cameraman_dog.jpg')
    save_image(edge_img_dog, f'{output_path}/cameraman_edges_dog.jpg')

def part_2_1(path, output_path='output/part2_1/', alpha=1.5):
    gaussian_1d = cv2.getGaussianKernel(ksize=9, sigma=2)
    gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)

    img = skio.imread(path) / 255.0 # , as_gray=True)

    # blur
    blurry_img = np.zeros_like(img)
    for c in range(img.shape[2]):
        blurry_img[:, :, c] = convolve2d(img[:, :, c], gaussian_kernel,
                                         mode='same', boundary='symm')

    # high freq sharpened image
    high_freq = img - blurry_img
    sharpened_img = img + alpha * high_freq
    sharpened_img = np.clip(sharpened_img, 0, 1)

    # unsharp mask filter
    unsharp_filter, gaussian_filter = utils.unsharp_mask_filter(sigma=1.5, alpha=alpha)
    sharpened_img_single = np.zeros_like(img)
    for c in range(img.shape[2]):
        sharpened_img_single[:, :, c] = convolve2d(img[:, :, c], unsharp_filter, mode='same', boundary='symm')
    sharpened_img_single = np.clip(sharpened_img_single, 0, 1)

    # display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.subplot(1, 4, 2)
    plt.imshow(blurry_img)
    plt.subplot(1, 4, 3)
    plt.imshow(high_freq)
    plt.subplot(1, 4, 4)
    plt.imshow(sharpened_img)

    plt.savefig(f'{output_path}/bird_sharpened_process.jpg', bbox_inches='tight')
    plt.show()

    save_image(sharpened_img, f'{output_path}/bird_sharpened.jpg')

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

part_1_1('data/self_portrait_2_cropped.jpg')
part_1_2('data/cameraman.png')
part_1_3('data/cameraman.png')
part_2_1('data/bird.jpg', alpha=1)

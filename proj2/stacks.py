import utils
from PIL import Image
import numpy as np
import skimage.io as skio
import skimage.util as skutil
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.transform import rescale
import cv2

def save_image(image, filename):
    if image.dtype in [np.float32, np.float64]:
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        im_out_uint8 = skutil.img_as_ubyte(image_normalized)
    else:
        im_out_uint8 = skutil.img_as_ubyte(image)
    
    skio.imsave(filename, im_out_uint8)

def convert_rgb(im):
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    return im

def get_gaussian(ksize=9, sigma=10):
    gaussian_1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    return gaussian_1d @ gaussian_1d.T

def gaussian_stack(im, levels, sigma):
    stack = [im.copy()]
    cpy = im.copy()

    for i in range(1, levels):
        size = sigma * (2 ** i)
        gaussian_kernel = get_gaussian(ksize=int(6 * size + 1), sigma=size)

        if im.ndim == 2: # grayscale
            blurry = convolve2d(cpy, gaussian_kernel, mode='same', boundary='fill')
        else: # color
            blurry = np.zeros_like(im)
            for channel in range(im.shape[2]):
                blurry[:, :, channel] = convolve2d(cpy[:, :, channel], gaussian_kernel, mode='same', boundary='fill')
        
        stack.append(blurry)
        cpy = blurry
    
    return stack

def laplacian_stack(im, levels, sigma):
    g_stack = gaussian_stack(im, levels, sigma)

    l_stack = []

    for i in range(len(g_stack) - 1):
        l_stack.append(g_stack[i] - g_stack[i + 1])
    
    l_stack.append(g_stack[-1]) # last level is gaussian stack

    return l_stack, g_stack

def create_mask(shape, direction='vertical', split=0.5):
    if len(shape) == 3:
        h, w, c = shape
        mask = np.zeros((h, w))
    else:
        h, w = shape
        mask = np.zeros((h, w))

    if direction == 'vertical':
        mask[:, :int(w * split)] = 1.0
    elif direction == 'horizontal':
        mask[:int(h * split), :] = 1.0
    elif direction == 'circle':
        center = (h // 4 + h // 2, w // 2 - w // 6)
        radius = int(min(h, w) // 4)
        cv2.circle(mask, center, radius, 1, -1)
    return mask

def multiresolution_blend(im1, im2, mask, levels=5, sigma=1):

    lap1, gaus1 = laplacian_stack(im1, levels, sigma)
    lap2, gaus2 = laplacian_stack(im2, levels, sigma)

    gaus = gaussian_stack(mask, levels, sigma)

    blend_stack = []
    laplacian1_masked = []
    laplacian2_masked = []
    for l1, l2, g in zip(lap1, lap2, gaus):
        g = np.clip(g, 0, 1)
        if len(im1.shape) == 3:
            g = np.stack([g] * 3, axis=2)
        
        masked_l1 = g * l1
        masked_l2 = (1 - g) * l2

        blend = masked_l1 + masked_l2 
        blend_stack.append(blend)
        laplacian1_masked.append(masked_l1)
        laplacian2_masked.append(masked_l2)

    # sum all levels
    blended = np.zeros_like(im1)
    for i in range(levels):
        blended += blend_stack[i]

    process_images = {
        'gaussian_1': gaus1,
        'gaussian_2': gaus2,
        'laplacian_1': lap1,
        'laplacian_2': lap2,
        'gaussian_mask': gaus,
        'laplacian_1_masked': laplacian1_masked,
        'laplacian_2_masked': laplacian2_masked,
        'blended': blended
    }

    return blended, process_images

def visualize_process(im1, im2, mask, blended, process_images, levels, output_path='output/part2_4/'):
    fig, axes = plt.subplots(5, levels + 1, figsize=(20, 20))
    axes[0, 0].imshow(im1)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(im2)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Mask')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(blended)
    axes[0, 3].set_title('Blended Result')
    axes[0, 3].axis('off')

    for i in range(4, levels + 1):
        axes[0, i].axis('off')

    for i in range(levels):
        axes[1, i].imshow(process_images['gaussian_1'][i], cmap='gray' if len(im1.shape) == 2 else None)
        axes[1, i].set_title(f'Gaussian 1, Level {i}')
        axes[1, i].axis('off')
    axes[1, levels].axis('off')

    for i in range(levels):
        axes[2, i].imshow(process_images['gaussian_2'][i], cmap='gray' if len(im2.shape) == 2 else None)
        axes[2, i].set_title(f'Gaussian 2, Level {i}')
        axes[2, i].axis('off')
    axes[2, levels].axis('off')

    for i in range(levels):
        lap = process_images['laplacian_1'][i]
        lap = lap + 0.5
        lap = np.clip(lap, 0, 1)
        axes[3, i].imshow(lap, cmap='gray' if len(im1.shape) == 2 else None)
        axes[3, i].set_title(f'Laplacian 1, Level {i}')
        axes[3, i].axis('off')
    axes[3, levels].axis('off')

    for i in range(levels):
        lap = process_images['laplacian_2'][i]
        lap = lap + 0.5
        lap = np.clip(lap, 0, 1)
        axes[4, i].imshow(lap, cmap='gray' if len(im2.shape) == 2 else None)
        axes[4, i].set_title(f'Laplacian 2, Level {i}')
        axes[4, i].axis('off')
    axes[4, levels].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_path}/oraple_blending_process.jpg', bbox_inches='tight')
    plt.show()

def visualize_process_fig_10(image1, image2, blended, process_images, levels, output_path='output/part2_4/'):
    fig, axes = plt.subplots(levels + 1, 3, figsize=(15, 20))

    display_levels = range(levels) 
    level_names = [str(lvl) for lvl in range(levels)] 
    
    for row, (level_idx, level_name) in enumerate(zip(display_levels, level_names)):
        # img 1
        lap_1_masked = process_images['laplacian_1_masked'][level_idx]
        lap_display = lap_1_masked + 0.5
        lap_display = np.clip(lap_display, 0, 1)
        
        axes[level_idx, 0].imshow(lap_display)
        axes[level_idx, 0].set_title(f'Laplacian 1, Level {level_name}')
        axes[level_idx, 0].axis('off')
        
        # img 2
        lap_2_masked = process_images['laplacian_2_masked'][level_idx]
        lap_display = lap_2_masked + 0.5
        lap_display = np.clip(lap_display, 0, 1)
        
        axes[level_idx, 1].imshow(lap_display)
        axes[level_idx, 1].set_title(f'Laplacian 2, Level {level_name}')
        axes[level_idx, 1].axis('off')
        
        # combined
        combined = lap_1_masked + lap_2_masked
        lap_display = combined + 0.5
        lap_display = np.clip(lap_display, 0, 1)
        
        axes[level_idx, 2].imshow(lap_display)
        axes[level_idx, 2].set_title(f'Combined Laplacian, Level {level_name}')
        axes[level_idx, 2].axis('off')
    
    # final results
    axes[levels, 0].imshow(image1)
    axes[levels, 0].set_title('Original Image 1')
    axes[levels, 0].axis('off')
    
    axes[levels, 1].imshow(image2)
    axes[levels, 1].set_title('Original Image 2')
    axes[levels, 1].axis('off')
    
    axes[levels, 2].imshow(blended)
    axes[levels, 2].set_title('Final Blended Result')
    axes[levels, 2].axis('off')
    
    plt.suptitle('Multiresolution Blending Process (Similar to Burt & Adelson (1983) Figure 10)\n\n', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_path}/oraple_blending_process_fig_10.jpg', bbox_inches='tight')
    plt.show()

im1 = skio.imread('data/apple.jpeg') / 255.0
im2 = skio.imread('data/orange.jpeg') / 255.0

im1 = convert_rgb(im1)
im2 = convert_rgb(im2)

im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

# im1 = rescale(im1, 0.25, channel_axis=-1, anti_aliasing=True)
# im2 = rescale(im2, 0.25, channel_axis=-1, anti_aliasing=True)

print(f"Resized images to: {im1.shape}, {im2.shape}")

mask = create_mask(im1.shape, 'vertical', 0.5)

levels = 6
sigma = 1.0

print(f"Running multiresolution blending with {levels} levels and sigma={sigma}")

blended, process_images = multiresolution_blend(im1, im2, mask, levels, sigma)

output_path = 'output/part2_4/'
save_image(blended, f'{output_path}/oraple.jpg')

visualize_process(im1, im2, mask, blended, process_images, levels, output_path)
visualize_process_fig_10(im1, im2, blended, process_images, levels, output_path)
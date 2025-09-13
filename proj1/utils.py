import skimage as sk
import numpy as np

def crop(img, percent=0.1):
    h, w = img.shape
    h_crop = int(h * percent)
    w_crop = int(w * percent)
    return img[h_crop:h - h_crop, w_crop:w - w_crop]

def gradient(img):
    gy, gx = np.gradient(img)
    return np.sqrt(gx**2 + gy**2)

def align(img, ref, shift=15, mode='l2'):
    loss = float('inf') if mode == 'l2' else -float('inf')
    displacement = (0, 0)

    # use gradient of images
    img_grad = gradient(img)
    ref_grad = gradient(ref)

    # crop to only center region to avoid boundary noise
    img_crop = crop(img_grad, 0.2)
    ref_crop = crop(ref_grad, 0.2)

    ref_mean = np.mean(ref_crop)
    ref_std = np.std(ref_crop)
    ref_norm = (ref_crop - ref_mean) / (ref_std + 1e-8)
    
    for i in range(-shift, shift + 1):
        for j in range(-shift, shift + 1):
            rolled = np.roll(np.roll(img_crop, i, axis=0), j, axis=1)

            # normalize the rolled image
            rolled_mean = np.mean(rolled)
            rolled_std = np.std(rolled)
            rolled_norm = (rolled - rolled_mean) / (rolled_std + 1e-8)

            if mode == 'l2':
                # calculate euclidean distance
                curr_loss = np.sqrt(np.sum((rolled_norm - ref_norm) ** 2))
                
                if curr_loss < loss:
                    loss = curr_loss
                    displacement = (i, j)

            elif mode == 'ncc':
                # calculate normalized cross-correlation w/ dot product
                curr_loss = np.sum(rolled_norm * ref_norm) / rolled_norm.size
            
                if curr_loss > loss:
                    loss = curr_loss
                    displacement = (i, j)
    
    new_img = np.roll(np.roll(img, displacement[0], axis=0), displacement[1], axis=1)
    return new_img, displacement, loss

def align_pyramid(img, ref, shift=15, scale_factor=0.5, mode='l2'):
    
    if max(img.shape) < 1000:
        return align(img, ref, shift)

    downscaled_img = sk.transform.rescale(img, scale_factor, anti_aliasing=True)
    downscaled_ref = sk.transform.rescale(ref, scale_factor, anti_aliasing=True)

    _, displacement, _ = align_pyramid(downscaled_img, downscaled_ref, shift, scale_factor)

    # scale displacement by inverse of scale_factor
    zoom_factor = 1 / scale_factor
    displacement = (int(displacement[0] * zoom_factor), int(displacement[1] * zoom_factor))

    loss = float('inf') if mode == 'l2' else float('-inf')

    img_grad = gradient(img)
    ref_grad = gradient(ref)

    img_crop = crop(img_grad, 0.3)
    ref_crop = crop(ref_grad, 0.3)

    ref_mean = np.mean(ref_crop)
    ref_std = np.std(ref_crop)
    ref_norm = (ref_crop - ref_mean) / (ref_std + 1e-8)

    # can search in smaller window after inital alignment for speedup
    dshift = 2
    for i in range(-dshift, dshift + 1):
        for j in range(-dshift, dshift + 1):
            di = displacement[0] + i
            dj = displacement[1] + j
            rolled = np.roll(np.roll(img_crop, di, axis=0), dj, axis=1)

            # normalize the rolled image
            rolled_mean = np.mean(rolled)
            rolled_std = np.std(rolled)
            rolled_norm = (rolled - rolled_mean) / (rolled_std + 1e-8)

            if mode == 'l2':
                # calculate euclidean distance
                curr_loss = np.sqrt(np.sum((rolled_norm - ref_norm) ** 2))
                
                if curr_loss < loss:
                    loss = curr_loss
                    displacement = (di, dj)

            elif mode == 'ncc':
                # calculate normalized cross-correlation w/ dot product
                curr_loss = np.sum(rolled_norm * ref_norm) / rolled_norm.size
            
                if curr_loss > loss:
                    loss = curr_loss
                    displacement = (di, dj)
    
    new_img = np.roll(np.roll(img, displacement[0], axis=0), displacement[1], axis=1)
    return new_img, displacement, loss
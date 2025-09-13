# CS180 (CS280A): Project 1

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.util as skutil
from utils import align, align_pyramid

def colorize_skel(imname, extension='tif'):
    # read in the image
    im = skio.imread(f'data/{imname}.{extension}')

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    # ag, g_displacement, g_loss = align(g, b, 20, mode='l2')
    # ar, r_displacement, r_loss = align(r, b, 20, mode='l2')

    ag, g_displacement, g_loss = align_pyramid(g, b, 50, mode='ncc')
    ar, r_displacement, r_loss = align_pyramid(r, b, 50, mode='ncc')


    print(imname)
    print(f'green displacement: {g_displacement}, loss: {g_loss}')
    print(f'red displacement: {r_displacement}, loss: {r_loss}')

    # create a color image
    im_out = np.dstack([ar, ag, b])
    im_out_uint8 = skutil.img_as_ubyte(im_out)

    # save the image
    fname = f'output/grad_ncc/{imname}_grad_ncc.jpg'
    skio.imsave(fname, im_out_uint8)

    # display the image
    skio.imshow(im_out)
    # skio.show()

# name of the input files
jpg_names = ['cathedral', 'monastery', 'tobolsk']
tif_names = ['church', 'emir', 'harvesters', 'icon', 
             'italil', 'lastochikino', 'lugano', 'melons', 
             'self_portrait', 'siren', 'three_generations']
custom_names = ['napoleon', 'vmalorossii', 'voranzhereie']

for name in jpg_names:
    colorize_skel(name, extension='jpg')

for name in tif_names:
    colorize_skel(name, extension='tif')

for name in custom_names:
    colorize_skel(name, extension='tif')
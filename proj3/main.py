import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import distance_transform_edt
import skimage.io as skio
import skimage.util as skutil

def save_image(image, filename):
    if image.dtype in [np.float32, np.float64]:
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        im_out_uint8 = skutil.img_as_ubyte(image_normalized)
    else:
        im_out_uint8 = skutil.img_as_ubyte(image)
    
    skio.imsave(filename, im_out_uint8)

def computeH(im1_pts, im2_pts):
    '''
    im1_pts, im2_pts are nx2 matrices
    '''
    n = im1_pts.shape[0]
    
    A = []
    for i in range(n):
        x1, y1 = im1_pts[i]
        x2, y2 = im2_pts[i]
        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2])
        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2])
    
    A = np.array(A)

    b = np.empty(2*n, dtype=np.float64)
    b[0::2] = im2_pts[:, 0] # x'
    b[1::2] = im2_pts[:, 1] # y'
    
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    H = np.vstack([h.reshape(8, 1), 1]).reshape(3, 3)
    
    return A, b, H, h

def computeBounds(im, H):
    h, w = im.shape[:2]
    
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1]
    ]).T
    
    trans = H @ corners
    # convert from homogenous to cartesian
    trans = trans / trans[2, :]

    x_min, x_max = trans[0, :].min(), trans[0, :].max()
    y_min, y_max = trans[1, :].min(), trans[1, :].max()
    
    return x_min, x_max, y_min, y_max

def warpImageNearestNeighbor(im, H):
    h, w = im.shape[:2]
    color = len(im.shape) == 3
    
    x_min, x_max, y_min, y_max = computeBounds(im, H)
    
    # output dimensions
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))
    
    if color:
        warped = np.zeros((out_h, out_w, im.shape[2]), dtype=im.dtype)
    else:
        warped = np.zeros((out_h, out_w), dtype=im.dtype)
    
    alpha = np.zeros((out_h, out_w), dtype=np.float32)
    
    H_inv = np.linalg.inv(H)
    
    # offset
    x_out, y_out = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x_out_shifted = x_out + x_min
    y_out_shifted = y_out + y_min
    
    ones = np.ones_like(x_out_shifted)
    coords_out = np.stack([x_out_shifted, y_out_shifted, ones], axis=0) # 3 x out_h x out_w

    # flatten
    coords_out = coords_out.reshape(3, -1) # 3 x (out_h * out_w)
    
    coords_in = H_inv @ coords_out # 3 x (out_h * out_w)
    coords_in = coords_in / coords_in[2, :]
    
    x_in = coords_in[0, :].reshape(out_h, out_w)
    y_in = coords_in[1, :].reshape(out_h, out_w)
    
    # compute NN
    x_in_nn = np.round(x_in).astype(int)
    y_in_nn = np.round(y_in).astype(int)
    
    valid = (x_in_nn >= 0) & (x_in_nn < w) & (y_in_nn >= 0) & (y_in_nn < h)
    
    if color:
        for c in range(im.shape[2]):
            warped[:, :, c][valid] = im[y_in_nn[valid], x_in_nn[valid], c]
    else:
        warped[valid] = im[y_in_nn[valid], x_in_nn[valid]]
    
    alpha[valid] = 1.0 # 1 if pixel is non-zero, else 0
    
    offset = (x_min, y_min)
    return warped, alpha, offset

def warpImageBilinear(im, H):
    h, w = im.shape[:2]
    color = len(im.shape) == 3
    
    x_min, x_max, y_min, y_max = computeBounds(im, H)
    
    # output dimensions
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))
    
    if color:
        warped = np.zeros((out_h, out_w, im.shape[2]), dtype=im.dtype)
    else:
        warped = np.zeros((out_h, out_w), dtype=im.dtype)
    
    alpha = np.zeros((out_h, out_w), dtype=im.dtype)
    
    H_inv = np.linalg.inv(H)
    
    # offset
    x_out, y_out = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x_out_shifted = x_out + x_min
    y_out_shifted = y_out + y_min
    
    ones = np.ones_like(x_out_shifted)
    coords_out = np.stack([x_out_shifted, y_out_shifted, ones], axis=0) # 3 x out_h x out_w

    # flatten
    coords_out = coords_out.reshape(3, -1) # 3 x (out_h * out_w)
    
    coords_in = H_inv @ coords_out # 3 x (out_h * out_w)
    coords_in = coords_in / coords_in[2, :]
    
    x_in = coords_in[0, :].reshape(out_h, out_w)
    y_in = coords_in[1, :].reshape(out_h, out_w)
    
    # bilinear interpolation
    x_floor = np.floor(x_in).astype(int)
    y_floor = np.floor(y_in).astype(int)
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    
    dx = x_in - x_floor
    dy = y_in - y_floor
    valid = (x_floor >= 0) & (x_ceil < w) & (y_floor >= 0) & (y_ceil < h)
    
    # calculate weights based on distance
    w_tl = (1 - dx) * (1 - dy)
    w_tr = dx * (1 - dy)
    w_bl = (1 - dx) * dy
    w_br = dx * dy
    
    if color:
        for c in range(im.shape[2]):
            warped[:, :, c][valid] = (
                w_tl[valid] * im[y_floor[valid], x_floor[valid], c] +
                w_tr[valid] * im[y_floor[valid], x_ceil[valid], c] +
                w_bl[valid] * im[y_ceil[valid], x_floor[valid], c] +
                w_br[valid] * im[y_ceil[valid], x_ceil[valid], c]
            )
    else:
        warped[valid] = (
            w_tl[valid] * im[y_floor[valid], x_floor[valid]] +
            w_tr[valid] * im[y_floor[valid], x_ceil[valid]] +
            w_bl[valid] * im[y_ceil[valid], x_floor[valid]] +
            w_br[valid] * im[y_ceil[valid], x_ceil[valid]]
        )
    
    alpha[valid] = 1.0 # 1 if pixel is non-zero, else 0
    
    if im.dtype == np.uint8:
        warped = np.clip(warped, 0, 255).astype(np.uint8)
    
    offset = (x_min, y_min)
    return warped, alpha, offset

def calculateDistWeight(alpha):
    # calculates distance to nearest zero pixel
    dist = distance_transform_edt(alpha)
    
    # normalize
    if dist.max() > 0:
        dist = dist / dist.max()
    
    return dist

def blendImages(im1, alpha1, offset1, im2, alpha2, offset2):
    # calculate mosaic size
    x_min = min(offset1[0], offset2[0])
    y_min = min(offset1[1], offset2[1])
    
    x_max = max(offset1[0] + im1.shape[1], offset2[0] + im2.shape[1])
    y_max = max(offset1[1] + im1.shape[0], offset2[1] + im2.shape[0])
    
    mosaic_w = int(np.ceil(x_max - x_min))
    mosaic_h = int(np.ceil(y_max - y_min))
    
    # create empty mosaic
    color = len(im1.shape) == 3
    if color:
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.float32)
    else:
        mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
    # represents weight of pixel from both images
    weight_sum = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
    
    # image 1
    x1_start = int(offset1[0] - x_min)
    y1_start = int(offset1[1] - y_min)
    x1_end = x1_start + im1.shape[1]
    y1_end = y1_start + im1.shape[0]
    
    weight1 = calculateDistWeight(alpha1)
    
    if color:
        for c in range(3):
            mosaic[y1_start:y1_end, x1_start:x1_end, c] += im1[:, :, c] * weight1
    else:
        mosaic[y1_start:y1_end, x1_start:x1_end] += im1 * weight1
    weight_sum[y1_start:y1_end, x1_start:x1_end] += weight1
    
    # image 2
    x2_start = int(offset2[0] - x_min)
    y2_start = int(offset2[1] - y_min)
    x2_end = x2_start + im2.shape[1]
    y2_end = y2_start + im2.shape[0]
    
    weight2 = calculateDistWeight(alpha2)
    
    if color:
        for c in range(3):
            mosaic[y2_start:y2_end, x2_start:x2_end, c] += im2[:, :, c] * weight2
    else:
        mosaic[y2_start:y2_end, x2_start:x2_end] += im2 * weight2
    weight_sum[y2_start:y2_end, x2_start:x2_end] += weight2
    
    # normalize pixels using weight sum
    valid = weight_sum > 0
    if color:
        for c in range(3):
            mosaic[:, :, c][valid] /= weight_sum[valid]
    else:
        mosaic[valid] /= weight_sum[valid]
    
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    return mosaic, (x_min, y_min)

def visualizeCorrespondences(im1, im2, im1_pts, im2_pts, title="Point Correspondences", name=""):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    H = max(h1, h2)
    combined = np.ones((H, w1 + w2, 3), dtype=np.uint8) * 255
    combined[:h1, :w1, :] = im1
    combined[:h2, w1:w1 + w2, :] = im2

    im2_pts_shifted = im2_pts + np.array([w1, 0])

    plt.figure(figsize=(15, 8))
    plt.imshow(combined)
    plt.axis('off')

    for i in range(im1_pts.shape[0]):
        p1 = im1_pts[i]
        p2 = im2_pts_shifted[i]
        color = np.random.rand(3,)

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, linewidth=2)
        plt.plot(p1[0], p1[1], 'o', color=color, markersize=7)
        plt.plot(p2[0], p2[1], 'x', color=color, markersize=8)

        plt.text(p1[0], p1[1]-20, str(i+1), color='white', fontsize=12, 
                ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        plt.text(p2[0], p2[1]-20, str(i+1), color='white', fontsize=12, 
                ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'output/correspondence_{name}.jpg', bbox_inches='tight')
    plt.show()

def visualizeWarping(im1_rgb, im2_rgb, im1_warped_nn, im1_warped_bil, name=""):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].imshow(im1_rgb)
    axes[0, 0].set_title("Image 1", fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(im2_rgb)
    axes[0, 1].set_title("Image 2", fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(im1_warped_nn)
    axes[1, 0].set_title("Warped (Nearest Neighbor)", fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(im1_warped_bil)
    axes[1, 1].set_title("Warped (Bilinear Interpolation)", fontsize=14)
    axes[1, 1].axis('off')

    save_image(im1_warped_nn, f'output/warping_nn_{name}.jpg')
    save_image(im1_warped_bil, f'output/warping_bilinear_{name}.jpg')
    
    plt.tight_layout()
    plt.savefig(f'output/warping_comparison_{name}.jpg', bbox_inches='tight')
    plt.show()

def visualizeRectification(im_rect_rgb, rect_cropped, rect_pts, name=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(im_rect_rgb)
    axes[0].plot(rect_pts[:, 0], rect_pts[:, 1], 'ro-', linewidth=3, markersize=12)
    for i, (x, y) in enumerate(rect_pts):
        axes[0].text(x, y-60, str(i+1), color='yellow', fontsize=16, ha='center',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(rect_cropped)
    axes[1].set_title("Rectified", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'output/rectification_{name}.jpg', bbox_inches='tight')
    plt.show()

def visualizeMosaic(im1_rgb, im2_rgb, im1_warped_bil, mosaic, name=""):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].imshow(im1_rgb)
    axes[0, 0].set_title("Image 1", fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(im2_rgb)
    axes[0, 1].set_title("Image 2", fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(im1_warped_bil)
    axes[1, 0].set_title("Warped Bilinear Interpolation (Image 1 → Image 2)", fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mosaic)
    axes[1, 1].set_title("Mosaic", fontsize=14)
    axes[1, 1].axis('off')

    save_image(mosaic, f'output/{name}.jpg')
    
    plt.tight_layout()
    plt.savefig(f'output/mosaic_{name}.jpg', bbox_inches='tight')
    plt.show()

def printSystem(A, b, H, h, name=""):
    n = im1_pts.shape[0]
    print(f"{name} System")
    print(f"System of equations for {n} point correspondences: Ah = b")
    print(f"\nMatrix A ({A.shape[0]} × {A.shape[1]}):")
    print(A)

    print(f"\nVector b ({b.shape[0]} × 1):")
    print(b)

    print(f"\nVector h ({h.shape[0]} x 1):")
    print(h)

    print(f"\nRecovered Homography Matrix H ({H.shape[0]} x {H.shape[1]}):")
    print(H)

if __name__ == "__main__":
    # A.1: LOAD IMAGES
    IM1_NAME = "data/garage1.jpeg"
    IM2_NAME = "data/garage2.jpeg"
    IM_RECT_NAME = "data/window.jpeg"

    NAME = "garage"
    RECT_NAME = "window"
    
    im1 = cv2.imread(IM1_NAME)
    im2 = cv2.imread(IM2_NAME)
    im_rect = cv2.imread(IM_RECT_NAME)
    
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im_rect_rgb = cv2.cvtColor(im_rect, cv2.COLOR_BGR2RGB)
    
    print(f"Image 1 ({NAME}): {im1_rgb.shape}")
    print(f"Image 2 ({NAME}): {im2_rgb.shape}")
    print(f"Rectification image ({RECT_NAME}): {im_rect_rgb.shape}")
    
    # A.2: RECOVER HOMOGRAPHIES
    # Correspondences: Image 1 -> Image 2
    # im1_pts = np.array([
    #     [1543, 2833], [2735, 2865], [1975, 455], [2123, 2549],
    #     [1713, 1831], [1893, 2247], [4635, 1645], [5153, 2827], [4023, 3137]
    # ])
    # path
    # im1_pts = np.array([
    #     [2127,2555],[1814,1998],[2907,2863],[2302,3057],[1900,2481],
    #     [3812,2171],[5110,2884],[3585,2863],[2224,3763],[3032,458],
    #     [1343,726],[2600,893],[1763,1839],[2388,1392],[3345,1413],
    #     [3450,787],[3337,2289],[4062,2600]
    # ])
    # bancroft
    # im1_pts = np.array([
    #     [1333,1882],[1227,3718],[1778,2038],[2360,994],[2009,3409],
    #     [2475,2795],[1389,2625],[764,3237],[2882,2212],[1783,2544],
    #     [1681,2181],[1377,2060],[2174,3993],[1705,3274],[1150,3083],[918,2829]
    # ])
    # haas
    # im1_pts = np.array([
    #     [1185,343],[1674,1301],[1701,827],[1325,1681],[2458,826],
    #     [1959,2488],[1928,1736],[3584,1194],[3337,861],[3492,1841],
    #     [3260,2128],[3863,1393],[3922,2259],[3932,2122],[3803,1737],
    #     [3643,1682],[3364,1452],[3003,532],[2631,312],[3229,2869],
    #     [2806,2984]
    # ])
    # room
    # im1_pts = np.array([
    #     [395,190],[372,479],[379,674],[581,188],[586,607],[757,189],
    #     [766,599],[933,192],[492,243],[817,71],[485,72],[494,563],
    #     [552,620],[605,523],[911,436],[871,531]
    # ])
    # chandelier
    # im1_pts = np.array([
    #     [262,667],[292,462],[503,424],[645,408],[501,665],[250,871],
    #     [136,686],[142,511],[168,373],[326,305],[287,973],[403,639],
    #     [468,610],[177,763],[179,581],[29,913]
    # ])
    # garage
    im1_pts = np.array([
        [558,561],[536,358],[558,351],[858,302],[886,396],[902,457],
        [926,510],[682,425],[826,554],[497,499],[404,437],[472,419],
        [399,280],[932,292],[797,462],[746,514],[664,319],[633,420],
        [671,544],[741,624],[608,547],[841,478],[457,543]
    ])
    
    # im2_pts = np.array([
    #     [421, 3005], [1759, 3019], [921, 457], [1065, 2697],
    #     [627, 1889], [823, 2363], [3547, 1817], [4009, 2885], [3041, 3219]
    # ])
    # path
    # im2_pts = np.array([
    #     [1079,2703],[741,2088],[1946,3002],[1274,3239],[827,2623],
    #     [2861,2287],[3975,2944],[2613,2974],[1154,4007],[2044,569],
    #     [121,700],[1534,975],[692,1904],[1317,1448],[2353,1537],
    #     [2489,945],[2410,2403],[3080,2699]
    # ])
    # bancroft
    # im2_pts = np.array([
    #     [783,1884],[665,3782],[1249,2047],[1804,1034],[1468,3404],
    #     [1912,2767],[846,2654],[157,3334],[2268,2205],[1255,2554],
    #     [1155,2193],[837,2068],[1633,3962],[1178,3290],[586,3143],[327,2892]
    # ])
    # haas
    # im2_pts = np.array([
    #     [18,12],[581,1170],[646,646],[102,1581],[1504,763],[849,2475],
    #     [866,1668],[2493,1234],[2308,907],[2384,1804],[2179,2063],
    #     [2696,1425],[2693,2175],[2712,2056],[2636,1720],[2516,1666],
    #     [2300,1450],[2038,557],[1696,272],[2096,2759],[1695,2909]
    # ])
    # room
    # im2_pts = np.array([
    #     [179,172],[140,485],[133,702],[386,191],[373,613],[551,209],
    #     [546,594],[697,228],[291,241],[604,105],[286,61],[278,575],
    #     [334,630],[393,527],[677,444],[642,527]
    # ])
    # chandelier
    # im2_pts = np.array([
    #     [251,437],[273,220],[504,176],[661,147],[485,437],[249,616],
    #     [128,454],[117,275],[126,111],[304,25],[285,697],[392,410],
    #     [457,386],[175,525],[160,350],[51,654]
    # ])
    # garage
    im2_pts = np.array([
        [193,572],[178,356],[198,351],[498,311],[521,401],[533,459],
        [556,511],[321,427],[463,558],[132,499],[30,437],[108,420],
        [33,273],[566,302],[436,465],[386,516],[310,321],[276,423],
        [311,544],[380,628],[243,549],[477,482],[86,549]
    ])
    
    A, b, H, h = computeH(im1_pts, im2_pts)
    
    printSystem(A, b, H, h, "Mosaic Homography")
    
    visualizeCorrespondences(im1_rgb, im2_rgb, im1_pts, im2_pts, "Mosaic Correspondences (Image 1 → Image 2)", name=NAME)
    
    # A.3: WARP IMAGES
    print("Warping with Nearest Neighbor")
    im1_warped_nn, alpha1_nn, offset1_nn = warpImageNearestNeighbor(im1_rgb, H)
    
    print("Warping with Bilinear Interpolation")
    im1_warped_bil, alpha1_bil, offset1_bil = warpImageBilinear(im1_rgb, H)
    
    print(f"Warped image shape: {im1_warped_bil.shape}")
    print(f"Offset: {offset1_bil}")
    
    # Compare interpolation methods
    visualizeWarping(im1_rgb, im2_rgb, im1_warped_nn, im1_warped_bil, name=NAME)

    # A.3: RECTIFICATION
    # frame
    # rect_pts = np.array([
    #     [173,297], # top-left
    #     [575,164], # top-right
    #     [629,793], # bottom-right
    #     [114,829], # bottom-left
    # ])
    # window
    rect_pts = np.array([
        [254,27],[566,212],[581,1074],[244,1006]
    ])

    rect_w, rect_h = 1000, 1333 # 750, 1000
    rect_target = np.array([
        [0, 0],
        [rect_w - 1, 0],
        [rect_w - 1, rect_h - 1],
        [0, rect_h - 1],
    ], dtype=np.float64)
    
    A_rect, b_rect, H_rect, h_rect = computeH(rect_pts, rect_target)
    
    printSystem(A_rect, b_rect, H_rect, h_rect, "Rectification Homography")
    
    print("Warping Rectification with Bilinear Interpolation")
    rect_warped_bil, alpha_rect_bil, offset_rect_bil = warpImageBilinear(im_rect_rgb, H_rect)
    
    # crop to only rectified
    x_start = max(0, int(-offset_rect_bil[0]))
    y_start = max(0, int(-offset_rect_bil[1]))
    x_end = min(rect_warped_bil.shape[1], x_start + rect_w)
    y_end = min(rect_warped_bil.shape[0], y_start + rect_h)
    
    rect_cropped = rect_warped_bil[y_start:y_end, x_start:x_end]
    
    visualizeRectification(im_rect_rgb, rect_cropped, rect_pts, name=RECT_NAME)
    
    # A.4: BLEND IMAGES INTO MOSAIC
    # Image 2 stays at its original position and image 1 is warped to image 2's coordinate plane
    im2_alpha = np.ones((im2_rgb.shape[0], im2_rgb.shape[1]), dtype=np.float32)
    im2_offset = (0, 0)

    mosaic, mosaic_offset = blendImages(
        im1_warped_bil, alpha1_bil, offset1_bil,
        im2_rgb, im2_alpha, im2_offset
    )
    
    print(f"Mosaic shape: {mosaic.shape}")

    visualizeMosaic(im1_rgb, im2_rgb, im1_warped_bil, mosaic, name=NAME)
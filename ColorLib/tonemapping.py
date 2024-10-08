import numpy as np
from numba import cuda
import math

# M. Kim and J. Kautz,
# “Consistent tone reproduction,” 
# in Proceedings of Computer Graphics and Imaging (2008)

@cuda.jit
def _img_luminance(img, L):
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:
        L[x, y] = 0.2126 * img[x, y, 0] + 0.7152 * img[x, y, 1] + 0.0722 * img[x, y, 2]

@cuda.jit
def _img_log_luminance(L, L_log):
    x, y = cuda.grid(2)
    if x < L.shape[0] and y < L.shape[1]:
        L_log[x, y] = math.log(L[x, y] + 1e-6)

@cuda.jit
def _compute_w(L_log, w, mu, sigma):
    x, y = cuda.grid(2)
    if x < L_log.shape[0] and y < L_log.shape[1]:
        diff = L_log[x, y] - mu
        w[x, y] = math.exp(-0.5 * (diff * diff) / (sigma * sigma))

@cuda.jit
def _compute_k2(w, k_2, k_1):
    x, y = cuda.grid(2)
    if x < w.shape[0] and y < w.shape[1]:
        k_2[x, y] = (1 - k_1) * w[x, y] + k_1

@cuda.jit
def _compute_L1(L_log, L_1, k_2, efficiency, mu):
    x, y = cuda.grid(2)
    if x < L_log.shape[0] and y < L_log.shape[1]:
        L_1[x, y] = math.exp(efficiency * k_2[x, y] * (L_log[x, y] - mu) + mu)

@cuda.jit
def _normalize_L1(L_1, L_1min, L_1max):
    x, y = cuda.grid(2)
    if x < L_1.shape[0] and y < L_1.shape[1]:
        val = L_1[x, y]
        val = min(max(val, L_1min), L_1max)
        L_1[x, y] = (val - L_1min) / (L_1max - L_1min + 1e-6)

@cuda.jit
def _L0_positive(L_0):
    x, y = cuda.grid(2)
    if x < L_0.shape[0] and y < L_0.shape[1]:
        L_0[x, y] = max(L_0[x, y], 1e-6)

@cuda.jit
def _apply_rgb(rgb_image, L_0, L_1):
    x, y = cuda.grid(2)
    if x < rgb_image.shape[0] and y < rgb_image.shape[1]:
        factor = L_1[x, y] / L_0[x, y]
        for c in range(3):
            rgb_image[x, y, c] *= factor
            if rgb_image[x, y, c] < 0:
                rgb_image[x, y, c] = 0.0

@cuda.jit
def _linear_rgb_to_srgb(linear_rgb, srgb):
    x, y = cuda.grid(2)
    if x < linear_rgb.shape[0] and y < linear_rgb.shape[1]:
        for c in range(3):
            val = linear_rgb[x, y, c]
            val = min(max(val, 0.0), 1.0)
            if val <= 0.0031308:
                srgb_val = 12.92 * val
            else:
                srgb_val = 1.055 * math.pow(val, 1 / 2.4) - 0.055
            srgb[x, y, c] = min(max(srgb_val, 0.0), 1.0)

def kimkautz(rgb_image, display_max=300, display_min=0.3, details=3, efficiency=0.5, epsilon=1e-6):
    """
    ### WARNING: CUDA FUNCTION

    Tonemap the given linear RGB image, and outputs an sRGB image. Note that your image must be in H, W form.

    M. Kim and J. Kautz,
    “Consistent tone reproduction,” 
    in Proceedings of Computer Graphics and Imaging (2008)

    Args:
        rgb_image (np.array(H, W, (R, G, B))): Original image in linear RGB form.
        display_max (float): Maximum candelas/m2 the monitor can output (nits).
        display_min (float): Minimum candelas/m2 the monitor can output.
        details (float): Higher numbers results in loss of detail in bright areas.
        efficiency (float): Higher numbers decrase the contrast of the entire image.
        epsilon (float): Number used to ensure division by zero does not occur.

    Returns:
        np.ndarray: sRGB image in the same format as rgb_image.
    """


    # Setup
    rgb_image = rgb_image.astype(np.float32)
    height, width, channels = rgb_image.shape # We need the dimensions of the image to put it on the device

    rgb_image_device = cuda.to_device(rgb_image)

    L_0_device = cuda.device_array((height, width), dtype=np.float32)
    L_log_device = cuda.device_array((height, width), dtype=np.float32)
    L_1_device = cuda.device_array((height, width), dtype=np.float32)
    w_device = cuda.device_array((height, width), dtype=np.float32)
    k_2_device = cuda.device_array((height, width), dtype=np.float32)
    srgb_image_device = cuda.device_array((height, width, channels), dtype=np.float32)

    threadsperblock = (16, 16) # TODO: Experiment with profilers
    blockspergrid_x = int(math.ceil(height / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(width / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Find the luminance of the image
    _img_luminance[blockspergrid, threadsperblock](rgb_image_device, L_0_device)

    # Find the log luminance and mu
    _img_log_luminance[blockspergrid, threadsperblock](L_0_device, L_log_device)

    L_log = L_log_device.copy_to_host()
    mu = np.mean(L_log)

    L_smax = np.max(L_log)
    L_smin = np.min(L_log)

    L_dmax = math.log(display_max)
    L_dmin = math.log(display_min)

    d0 = L_smax - L_smin

    k_1 = (L_dmax - L_dmin) / (d0 + epsilon) # Scale factor

    sigma = d0 / details

    # Bulk computations
    _compute_w[blockspergrid, threadsperblock](L_log_device, w_device, mu, sigma)
    _compute_k2[blockspergrid, threadsperblock](w_device, k_2_device, k_1)
    _compute_L1[blockspergrid, threadsperblock](L_log_device, L_1_device, k_2_device, efficiency, mu)

    # Take final image
    L_1 = L_1_device.copy_to_host()
    L_1min = np.quantile(L_1, 0.01)
    L_1max = np.quantile(L_1, 0.99)

    # Make sure the image isn't broken
    _normalize_L1[blockspergrid, threadsperblock](L_1_device, L_1min, L_1max)
    _L0_positive[blockspergrid, threadsperblock](L_0_device)

    # Convert colorspace
    _apply_rgb[blockspergrid, threadsperblock](rgb_image_device, L_0_device, L_1_device)
    _linear_rgb_to_srgb[blockspergrid, threadsperblock](rgb_image_device, srgb_image_device)

    srgb_image = srgb_image_device.copy_to_host()

    return srgb_image

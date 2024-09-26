import numpy as np

M = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
])

M0 = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
])

def srgb(xyz):
    return np.dot(M, xyz)**2.2

def rgb_to_xyz(rgb):
    return np.dot(M0, rgb)

# M. Kim and J. Kautz,
# “Consistent tone reproduction,” 
# Proceedings of Computer Graphics and Imaging (2008)

def kimkautz(xyz, average_image_luminance, scaling_factor, log_dynamic_range, details=3, efficiency=0.5):
    log_luminance = np.log10(xyz[1])
    log_avg_luminance = np.log10(average_image_luminance)

    luminance_ratio = (log_luminance - log_avg_luminance)
    w = np.exp(-0.5 * ((luminance_ratio**2) / (log_dynamic_range/details)**2))
    non_linear_scalar = (1 - scaling_factor) * w + scaling_factor
    l1 = np.exp(efficiency * non_linear_scalar * luminance_ratio + log_avg_luminance)

    c = (l1/xyz[1])
    cx = c * xyz[0]
    cz = c * xyz[2]
    c1 = np.array([cx, l1, cz])

    srgb = np.dot(M, c1)**2.2

    return srgb


def kimkautz_scaling_factor(max_luminance, display_max, display_min):
    """Calculate the scaling factor and log dynamic range for the kimkautz tone mapper.
    
    Args:
        max_luminance: The maximum luminance in the image (Y componenet of CIE XYZ color space)
        display_max: The maximum luminance of the display in cd/m2 (nix)
        display_min: The minimum luminance of the display in cd/m2 (typically ~0.3 or lower)
        
    Raises:
        ValueError: If any parameter is less than or equal to zero.
        
    Returns: 
        A tuple of the scalar, and then the log dynamic range"""
    if max_luminance <= 0 or display_max <= 0 or display_min <= 0:
        raise ValueError("The scaling factor must not include parameters below or equal to 0!")
    
    scalar = np.log10(max_luminance) / (np.log10(display_max) - np.log10(display_min))
    dynamic_range = np.log10(display_max / display_min)
    return scalar, dynamic_range

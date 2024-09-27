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

def xyz_to_rgb(xyz):
    return np.dot(xyz, M.T)

def linear_rgb_to_srgb(linear_rgb):
    linear_rgb = np.asarray(linear_rgb, dtype=np.float64)
    linear_rgb = np.nan_to_num(linear_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    linear_rgb = np.clip(linear_rgb, 0, None)

    srgb = np.where(
        linear_rgb <= 0.0031308, 
        12.92 * linear_rgb, 
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055
    )
    
    return np.clip(srgb, 0, 1)

def rgb_to_xyz(rgb):
    return np.dot(rgb, M0.T)


def reinhard_tone_map(xyz):
    # Apply tone mapping to luminance
    L = xyz[1]
    L_d = L / (1 + L)
    # Reconstruct XYZ with tone-mapped luminance
    c = L_d / (L + 1e-6)
    xyz_tm = xyz * c
    # Convert back to RGB
    rgb = xyz_to_rgb(xyz_tm)
    # Apply gamma correction
    srgb = linear_rgb_to_srgb(rgb)
    return srgb


# M. Kim and J. Kautz,
# “Consistent tone reproduction,” 
# in Proceedings of Computer Graphics and Imaging (2008)

def kimkautz(xyz, average_image_luminance, scaling_factor, log_dynamic_range, details=3, efficiency=0.5, epsilon=1e-6):
    log_luminance = np.log10(xyz[1] + epsilon)
    log_avg_luminance = np.log10(average_image_luminance + epsilon)
    
    luminance_ratio = log_luminance - log_avg_luminance
    w = np.exp(-0.5 * ((luminance_ratio**2) / (log_dynamic_range/details)**2))
    non_linear_scalar = (1 - scaling_factor) * w + scaling_factor
    l1 = np.exp(efficiency * non_linear_scalar * luminance_ratio + log_avg_luminance)
    
    c = (l1 / (xyz[1] + epsilon))
    cx = c * xyz[0]
    cz = c * xyz[2]
    c1 = np.array([cx, l1, cz])
    
    rgb = xyz_to_rgb(c1)
    srgb = linear_rgb_to_srgb(rgb)
    
    return srgb

def linear_kimkautz(xyz, scaling_factor):
    log_luminance = np.log10(xyz[1] + 1e-6)
    l1 = np.exp(scaling_factor * log_luminance)

    c = (l1/(xyz[1] + 1e-6))
    cx = c * xyz[0]
    cz = c * xyz[2]
    c1 = np.array([cx, l1, cz])

    rgb = xyz_to_rgb(c1)
    srgb = linear_rgb_to_srgb(rgb)
    
    return srgb

def kimkautz_scaling_factor(max_luminance, min_luminance, display_max, display_min):
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
    
    image_dynamic_range = np.log10(max_luminance) - np.log10(min_luminance)
    display_dynamic_range = np.log10(display_max) - np.log10(display_min)
    
    # Prevent division by zero or extremely small scaling factors
    if image_dynamic_range == 0:
        image_dynamic_range = 1e-6
    
    scalar = display_dynamic_range / image_dynamic_range
    dynamic_range = display_dynamic_range
    return scalar, dynamic_range

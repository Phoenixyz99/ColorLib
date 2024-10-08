import numpy as np

# Chris Wyman, Peter-Pike Sloan, and Peter Shirley, 
# "Simple Analytic Approximations to the CIE XYZ Color Matching Functions", 
# in Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 1–11, 2013.

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
    """Converts a 3-long np.ndarray from CIE 1931 XYZ coordinates into RGB.
    
    Args:
        xyz (np.ndarray): Color in XYZ color space.
    
    Returns:
        np.ndarray: The color in RGB color space.
    """

    return np.dot(xyz, M.T)

def rgb_to_xyz(rgb):
    """Converts a 3-long np.ndarray from linear RGB to CIE 1931 XYZ.
    
    Args:
        xyz (np.ndarray): Color in XYZ color space.
    
    Returns:
        np.ndarray: The color in RGB color space.
    """

    return np.dot(rgb, M0.T)

def linear_to_srgb(rgb):
    """Converts linear RGB to sRGB.
    
    Args:
        rgb (np.ndarray): 3-long array of the color in RGB space.
        
    Returns:
        np.ndarray: 3-long array of the color in sRGB space.
    """
    
    def gamma(channel):
        if channel <= 0.0031308:
            return 12.92 * channel
        else:
            return 1.055 * (channel **(1.0 / 2.4)) - 0.055

    srgb = np.array([gamma(channel) for channel in rgb])
    srgb = np.clip(srgb, 0, 1)

    return srgb


def _gausian_fit(lamb, a, B, y, g):
    amount = y if lamb < B else g
    return (np.exp((((lamb - B) * amount)**2)/-2))*a

def wavelength_to_xyz(wavelength, illuminant=np.array([0.95047,1,1.08883])):
    """Converts a wavelength into its RGB counterpart using a piecewise fit.

    Chris Wyman, Peter-Pike Sloan, and Peter Shirley, 
    "Simple Analytic Approximations to the CIE XYZ Color Matching Functions", 
    in Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 1–11, 2013.

    Args:
        wavelength (float): Wavelength of light in nm.
        illuminant (np.array(3)): CIE illuminant. Defaults to D65.
        
    Raises:
        ValueError: If wavelength is less than or equal to zero.
        
    Returns:
        np.ndarray: The color in XYZ space (3-long numpy array).
    """

    if wavelength <= 0.0:
        raise ValueError("Wavelength must be greater than zero.")

    x = _gausian_fit(wavelength, 0.362, 442, 0.0624, 0.0374) + _gausian_fit(wavelength, 1.056, 599.8, 0.0264, 0.0323) + _gausian_fit(wavelength, -0.065, 501.1, 0.049, 0.0382)
    y = _gausian_fit(wavelength, 0.821, 568.8, 0.0213, 0.0247) + _gausian_fit(wavelength, 0.286, 530.9, 0.0613, 0.0322)
    z = _gausian_fit(wavelength, 1.217, 437, 0.0845, 0.0278) + _gausian_fit(wavelength, 0.681, 459, 0.0385, 0.0725)

    XYZ = ([x, y, z] * illuminant)

    return XYZ
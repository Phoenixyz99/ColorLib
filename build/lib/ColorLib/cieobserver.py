import numpy as np

# Chris Wyman, Peter-Pike Sloan, and Peter Shirley, 
# "Simple Analytic Approximations to the CIE XYZ Color Matching Functions", 
# Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 1–11, 2013.
# Available at: http://jcgt.org/published/0002/02/01/

def _gausian_fit(lamb, a, B, y, g):
    amount = y if lamb < B else g
    return (np.exp((((lamb - B) * amount)**2)/-2))*a

def wavelength_to_xyz(wavelength, illuminant=np.array([0.95047,1,1.08883])):
    """Converts a wavelength into its RGB counterpart using a piecewise fit.

    Args:
        wavelength: Wavelength of light in nm
        illuminant: CIE illuminant. Defaults to D65
        
    Raises:
        ValueError: If wavelength is less than or equal to zero.
        
    Returns: 
        The color in XYZ space (3-long numpy array)"""

    if wavelength <= 0.0:
        raise ValueError("Wavelength must be greater than zero.")

    x = _gausian_fit(wavelength, 0.362, 442, 0.0624, 0.0374) + _gausian_fit(wavelength, 1.056, 599.8, 0.0264, 0.0323) + _gausian_fit(wavelength, -0.065, 501.1, 0.049, 0.0382)
    y = _gausian_fit(wavelength, 0.821, 568.8, 0.0213, 0.0247) + _gausian_fit(wavelength, 0.286, 530.9, 0.0613, 0.0322)
    z = _gausian_fit(wavelength, 1.217, 437, 0.0845, 0.0278) + _gausian_fit(wavelength, 0.681, 459, 0.0385, 0.0725)

    XYZ = ([x, y, z] * illuminant)

    return XYZ
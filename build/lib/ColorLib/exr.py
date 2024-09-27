import os
import OpenEXR
import Imath
import numpy as np

def read_exr(file_path: str) -> np.ndarray:
    """Reads an EXR file at file_path and outputs a numpy array in the format [(H, W, (R, G, B))].

    Args:
        file_path (str): Path to the EXR file.

    Raises:
        FileNotFoundError: The specified file does not exist.
        Exception: Other errors.
    
    Returns:
        np.ndarray: Numpy array representing the EXR image in the format (W, H, (R, G, B)).
    """
    # Validate file path
    if not isinstance(file_path, str):
        raise ValueError(f"Expected a string for file_path, but got {type(file_path).__name__}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Open the EXR file
        exr_file = OpenEXR.InputFile(file_path)

        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Assuming RGB channels
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ['R', 'G', 'B']
        rgb_data = [np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32) for c in channels]
        rgb_data = [channel.reshape((width, height)) for channel in rgb_data]  # Reshape to (H, W)

        # Close the EXR file after reading
        exr_file.close()

        # Stacked to (W, H, (R, G, B))
        rgb_image = np.stack(rgb_data, axis=-1)

        return rgb_image
    
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

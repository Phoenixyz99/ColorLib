import os
import OpenEXR
import Imath
import numpy as np

def read_exr(file_path):
    """Reads an exr file and outputs it in the form of (H, W, (R, G, B)).

    Args:
        file_path (str): Location of the exr to read.

    Raises:
        FileNotFoundError: If the file path is invalid.
        ValueError: If the float/int type for pixel channels is invalid.
        RuntimeError: If the file is not in the expected EXR format or missing channels.
    
    Returns:
        np.ndarray or dict: A numpy array in the format of (H, W, (R, G, B)), 
        or a dict containing all channel data if RGB is not present.
    """

    
    # Check if the file exists before proceeding
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        exr_file = OpenEXR.InputFile(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read EXR file. Error: {e}")

    try:
        # Read the EXR file's header and data window
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Read channel data from the file
        channels = header['channels'].keys()
        channel_data = {}

        for channel in channels:
            pixel_type = header['channels'][channel].type

            # Identify the correct data type for the channel
            if pixel_type == Imath.PixelType(Imath.PixelType.HALF):
                dtype = np.float16
            elif pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                dtype = np.float32
            elif pixel_type == Imath.PixelType(Imath.PixelType.UINT):
                dtype = np.uint32
            else:
                raise ValueError(f"Unsupported pixel type: {pixel_type}")

            # Read and reshape the channel data
            channel_data[channel] = np.frombuffer(exr_file.channel(channel, pixel_type), dtype=dtype).reshape((height, width))

            # Convert 16-bit floats to 32-bit for consistency
            if dtype == np.float16:
                channel_data[channel] = channel_data[channel].astype(np.float32)

        # Ensure that R, G, B channels exist for RGB output
        if all(c in channels for c in ['R', 'G', 'B']):
            rgb_image = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=-1)
            rgb_image = np.clip(rgb_image, 0.0, None)  # Clip any negative values to 0
            return rgb_image
        else:
            # If not all RGB channels are present, return all available channel data
            return channel_data
    except Exception as e:
        raise RuntimeError(f"Error processing EXR file: {e}")
    finally:
        # Ensure that the file is closed after processing
        exr_file.close()

    

def save_exr_image(file_path, image_array):
    """
    Saves an EXR image from a NumPy array of shape (W, H, 3), where each element
    corresponds to a pixel's (R, G, B) values.
    
    Args:
        file_path (str): The output path (including name) for the EXR image.
        np.ndarray: An array containing pixel data in the shape (W, H, (R, G, B)).
    """

    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image must be a 3D NumPy array of shape (W, H, 3) representing RGB values.")

    height, width, _ = image_array.shape

    header = OpenEXR.Header(width, height)

    R = image_array[:, :, 0].astype(np.float32).tobytes()
    G = image_array[:, :, 1].astype(np.float32).tobytes()
    B = image_array[:, :, 2].astype(np.float32).tobytes()

    exr_file = OpenEXR.OutputFile(file_path, header)
    
    exr_file.writePixels({'R': R, 'G': G, 'B': B})

    exr_file.close()


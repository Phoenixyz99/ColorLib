import OpenEXR
import Imath
import numpy as np
import os
import matplotlib.pyplot as plt
import mplcursors
import ast
import re

plt.style.use('dark_background')


def read_exr(file_path):
    """Reads an EXR file and outputs the image data (H, W, (R, G, B)) or channel data, along with metadata.
    
    Args:
        file_path (str): Location of the EXR to read.
    
    Raises:
        FileNotFoundError: If the file path is invalid.
        ValueError: If the float/int type for pixel channels is invalid.
        RuntimeError: If the file is not in the expected EXR format or missing channels.
    
    Returns:
        np.ndarray: An array of the image with the shape (H, W, (R, G, B))
        dict: A dict of metadata including custom attributes.
    """
    import OpenEXR
    import Imath
    import numpy as np
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found! {file_path}")

    try:
        exr_file = OpenEXR.InputFile(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read EXR file! {e}")

    try:
        header = exr_file.header()
        data_window = header['dataWindow']
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1

        channels = header['channels'].keys()
        channel_data = {}

        # Extract standard metadata
        metadata = {
            'compression': str(header['compression']),
            'channels': list(channels),
            'width': width,
            'height': height,
            'pixel_aspect_ratio': header.get('pixelAspectRatio', 'N/A'),
            'line_order': str(header.get('lineOrder', 'N/A')),
            'display_window': header.get('displayWindow', 'N/A'),
            'data_window': header.get('dataWindow', 'N/A'),
            'screen_window_center': header.get('screenWindowCenter', 'N/A'),
            'screen_window_width': header.get('screenWindowWidth', 'N/A'),
        }

        # Extract custom metadata (all other attributes from the header)
        custom_metadata = {}
        for key, value in header.items():
            if key not in metadata:
                custom_metadata[key] = str(value)

        # Extract channel data
        for channel in channels:
            pixel_type = header['channels'][channel].type

            if pixel_type == Imath.PixelType(Imath.PixelType.HALF):
                dtype = np.float16
            elif pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                dtype = np.float32
            elif pixel_type == Imath.PixelType(Imath.PixelType.UINT):
                dtype = np.uint32
            else:
                raise ValueError(f"Unsupported pixel type: {pixel_type}")

            # Format in the shape (H, W)
            channel_data[channel] = np.frombuffer(exr_file.channel(channel, pixel_type), dtype=dtype).reshape((height, width))

            if dtype == np.float16:
                channel_data[channel] = channel_data[channel].astype(np.float32)

        # Check if R, G, B channels exist, and return the RGB image if present
        if all(c in channels for c in ['R', 'G', 'B']):
            rgb_image = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=-1)
            rgb_image = np.clip(rgb_image, 0.0, None)  # Clamp values to avoid negative values
            return rgb_image, metadata, custom_metadata
        else:
            # Return all channel data if it's not an RGB image
            return channel_data, metadata, custom_metadata

    except Exception as e:
        raise RuntimeError(f"Error processing EXR file! {e}")
    finally:
        exr_file.close()






def clean_value(value):
    """
    Cleans the metadata value by decoding actual bytes and string representations of bytes,
    and converting to appropriate types.
    """
    # If the value is a bytes object, decode it directly
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')
        except UnicodeDecodeError:
            value = value.decode('latin1')
    
    # Check if value is a string representation of bytes (e.g., "b'fast'")
    elif isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
        # Strip the "b'" and "'" and decode the content
        try:
            value = value[2:-1].encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            value = value[2:-1].encode('utf-8').decode('latin1')
    
    # If the value is now a string, attempt to evaluate it to a literal
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Keep value as is if it cannot be evaluated
    return value


def save_exr_image(file_path, image_array, metadata=None):
    """
    Saves an EXR image from a NumPy array of shape (H, W, 3), where each element
    corresponds to a pixel's (R, G, B) values and allows custom metadata to be added.
    """
    # Validate the input array shape
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image must be a numpy array of shape (H, W, 3)!")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract image dimensions
    height, width, _ = image_array.shape
    
    # Create an EXR header with image dimensions
    header = OpenEXR.Header(width, height)
    
    # Exclude problematic metadata keys
    excluded_keys = {'channels', 'compression', 'dataWindow', 'displayWindow', 'lineOrder'}
    
    # Add custom metadata if provided
    if metadata:
        for key, value in metadata.items():
            if key in excluded_keys:
                continue  # Skip keys that are handled by OpenEXR internally
            
            # Clean the value
            clean_val = clean_value(value)
            
            # After cleaning, check the type and assign accordingly
            if isinstance(clean_val, (int, float)):
                header[key] = clean_val  # Use the number directly
            elif isinstance(clean_val, str):
                header[key] = clean_val.encode('utf-8')  # Encode strings
            elif isinstance(clean_val, tuple):
                if all(isinstance(item, (int, float)) for item in clean_val):
                    if len(clean_val) == 2:
                        header[key] = Imath.V2f(*clean_val)
                    elif len(clean_val) == 3:
                        header[key] = Imath.V3f(*clean_val)
                    else:
                        raise ValueError(f"Unsupported tuple size for key {key}: {len(clean_val)}. Only tuples of size 2 or 3 are supported.")
                else:
                    raise ValueError(f"Unsupported tuple contents for key {key}. All elements must be int or float.")
            else:
                # For other types, convert to string and encode
                header[key] = str(clean_val).encode('utf-8')
    
    # Convert RGB channels to float32 and prepare for writing
    R = image_array[:, :, 0].astype(np.float32).tobytes()
    G = image_array[:, :, 1].astype(np.float32).tobytes()
    B = image_array[:, :, 2].astype(np.float32).tobytes()
    
    # Create and write to the EXR file
    exr_file = OpenEXR.OutputFile(str(file_path), header)
    exr_file.writePixels({'R': R, 'G': G, 'B': B})
    exr_file.close()




def plot(yaxis, xaxis=None, xlabel="X", ylabel="Y", scale="linear"):
    """Plots the given axis on a matplotlib graph, with a data cursor.
    
    Args:
        yaxis (np.ndarray, required): An array to plot.
        xaixs (np.ndarray): The axis to plot against. This must have the same length as the yaxis. 
         If no xaxis is given, the yaxis will be plotted against its index.
        xlabel (str): The title of the x axis.
        ylabel (str): The title of the y axis.
        scale (str): The scale/type of graph. "linear", "logy", "logx", "logxy", "polar", "logpolar".

    Returns:
        None
        """

    plt.figure(figsize=(10, 5))
    
    if xaxis is None:
        xaxis = list(range(len(yaxis))) # Use the index of yaxis rather than xaxis
    
    if scale == "logy":
        plt.yscale('log')
    elif scale == "logx":
        plt.xscale('log')
    elif scale == "logxy":
        plt.xscale('log')
        plt.yscale('log')
    elif scale == "polar":
        ax = plt.subplot(111, projection='polar')
        ax.plot(xaxis, yaxis, label='Item')
    elif scale == "logpolar":
        ax = plt.subplot(111, projection='polar')
        ax.set_yscale('log')
        ax.plot(xaxis, yaxis, label='Item')
    else:
        plt.plot(xaxis, yaxis, label='Item')

    plt.title(f"{xlabel} vs {ylabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    # Enable cursor
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f'X: {xaxis[int(sel.index)]:.16f}\n'
        f'Y: {yaxis[int(sel.index)]:.16f}\n'))

    plt.show()

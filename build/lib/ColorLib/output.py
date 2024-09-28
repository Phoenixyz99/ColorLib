import os
import imageio
import numpy as np

def read_exr(file_path):
    """
    Reads an EXR file and outputs it in the form of (H, W, (R, G, B)).

    Args:
        file_path (str): Location of the EXR to read.

    Raises:
        FileNotFoundError: If the file path is invalid.
        ValueError: If the image does not contain at least three channels.
        RuntimeError: If the file cannot be read properly.

    Returns:
        np.ndarray or dict: A NumPy array in the format of (H, W, (R, G, B)), 
        or a dict containing all channel data if RGB is not present.
    """

    # Check if the file exists before proceeding
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Read the EXR image using imageio
        exr_image = imageio.imread(file_path, format='EXR-FI')

        if exr_image is None:
            raise RuntimeError(f"Failed to read EXR file: {file_path}")

        # imageio may read EXR as (H, W, C)
        if len(exr_image.shape) == 3 and exr_image.shape[2] >= 3:
            # Assuming the channels are in RGB order
            rgb_image = exr_image[:, :, :3]

            # Clip any negative values to 0
            rgb_image = np.clip(rgb_image, 0.0, None)

            if exr_image.shape[2] > 3:
                # If there are more than three channels, return as a dict
                channels = {}
                for i in range(exr_image.shape[2]):
                    channels[f'Channel_{i}'] = exr_image[:, :, i]
                return channels

            return rgb_image
        elif len(exr_image.shape) == 2:
            # Single channel EXR
            return exr_image
        else:
            # Multiple channels but not RGB
            channels = {}
            for i in range(exr_image.shape[2]):
                channels[f'Channel_{i}'] = exr_image[:, :, i]
            return channels

    except Exception as e:
        raise RuntimeError(f"Error processing EXR file: {e}")

def save_exr_image(file_path, image_array):
    """
    Saves an EXR image from a NumPy array of shape (H, W, 3), where each element
    corresponds to a pixel's (R, G, B) values.

    Args:
        file_path (str): The output path (including name) for the EXR image.
        image_array (np.ndarray): An array containing pixel data in the shape (H, W, 3) representing RGB values.

    Raises:
        ValueError: If the input image does not have the correct shape or type.
        RuntimeError: If the image cannot be saved properly.
    """

    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image must be a 3D NumPy array of shape (H, W, 3) representing RGB values.")

    try:
        # imageio expects the image in RGB order
        # Ensure the image is in float32 format for EXR
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)

        # Clip any negative values to 0
        image_array = np.clip(image_array, 0.0, None)

        # Save the EXR image using imageio
        imageio.imwrite(file_path, image_array, format='EXR-FI')

    except Exception as e:
        raise RuntimeError(f"Failed to write EXR file: {file_path}. Error: {e}")

# Example Usage
if __name__ == "__main__":
    input_path = "input.exr"
    output_path = "output.exr"

    try:
        # Read the EXR image
        image = read_exr(input_path)
        print(f"Image shape: {image.shape}")

        # Perform any processing if needed
        # For example, normalize the image
        # image = image / np.max(image)

        # Save the EXR image
        save_exr_image(output_path, image)
        print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

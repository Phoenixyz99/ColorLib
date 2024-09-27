import os
import OpenEXR
import Imath
import numpy as np

def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)

    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = header['channels'].keys()

    channel_data = {}

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

        channel_data[channel] = np.frombuffer(exr_file.channel(channel, pixel_type), dtype=dtype).reshape((height, width))

        if dtype == np.float16:
            channel_data[channel] = channel_data[channel].astype(np.float32)

    if all(c in channels for c in ['R', 'G', 'B']):
        rgb_image = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=-1)
        rgb_image = np.clip(rgb_image, 0.0, None)
        return rgb_image
    else:
        return channel_data


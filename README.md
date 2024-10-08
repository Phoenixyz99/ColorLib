# ColorLib
 A python library that provides common functionality related to color spaces and EXR image generation.
 
## Installation
 
 Install from source:
 ```
 git clone https://github.com/Phoenixyz99/ColorLib.git
 cd ColorLib
 pip install .
 ```

 To update once installed:
 ```
 cd ColorLib
 git pull origin main
 ```

## Notice
 The tonemapper included in this plugin uses numba to run CUDA-accelerated operations to allow for real-time tone mapping.
 These functions will not run on a device without a CUDA-enabled GPU.

This project would not be possible without the following research:

Chris Wyman, Peter-Pike Sloan, and Peter Shirley, 
"Simple Analytic Approximations to the CIE XYZ Color Matching Functions", 
in Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 1–11, 2013.
Copyright (c) 2013 Chris Wyman, Peter-Pike Sloan, and Peter Shirley 

M. Kim and J. Kautz,
“Consistent tone reproduction,” 
in Proceedings of Computer Graphics and Imaging (2008)

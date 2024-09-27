from setuptools import setup, find_packages

setup(
    name="ColorLib",
    version="0.1.0",
    author="Phoenix Allen",
    author_email="Phoenxyz99@gmail.com",
    description="A library containing color space and EXR file generation functions.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Phoenixyz99/ColorLib",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",   # Adding NumPy dependency
        "numba>=0.53.0",   # Adding Numba (supports CUDA)
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: POSIX :: Other",
    ],
    python_requires='>=3.6',
)

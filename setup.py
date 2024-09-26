from setuptools import setup, find_packages

setup(
    name="ColorLib",
    version="0.1.0",
    author="Phoenix Allen",
    author_email="Phoenxyz99@gmail.com",
    description="A library containing color space and EXR file generation functions.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/my_shared_library",
    packages=find_packages(),
    install_requires=[
        # List any dependencies here, e.g.,
        # "requests>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT Licernse",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ColorLib",
    version="0.1.0",
    author="Phoenix Allen",
    author_email="phoenxyz99@gmail.com",
    description="A library containing color space and EXR file generation functions.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Phoenixyz99/ColorLib",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "numba>=0.53.0",
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

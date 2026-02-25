"""Setup script for mrst-python package."""

from setuptools import setup, find_packages

setup(
    name="mrst",
    version="0.1.0",
    description="Python interface to MRST (MATLAB Reservoir Simulation Toolbox) via Octave",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MRST Contributors",
    license="GPLv3",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "oct2py>=5.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.4", "pyvista>=0.36"],
        "data": ["pandas>=1.3"],
        "all": ["matplotlib>=3.4", "pyvista>=0.36", "pandas>=1.3"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)

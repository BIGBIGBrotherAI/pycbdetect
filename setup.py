#!/usr/bin/env python3
"""Installation script for pycbdetect."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pycbdetect",
    version="0.1.0",
    author="Andreas Geiger (original MATLAB), ftdlyc (C++ port)",
    description=("Pure-Python checkerboard / deltille pattern detection"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ftdlyc/libcbdetect",
    license="GPL-3.0-or-later",
    python_requires=">=3.9",
    packages=find_packages(include=["pycbdetect*", "pycbdetect.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.5",
    ],
)

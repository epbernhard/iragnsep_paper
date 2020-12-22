#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iragnsep", # Replace with your own username
    version="7.2.0",
    author="Emmanuel Bernhard",
    author_email="manu.p.bernhard@gmail.com",
    description="Fits of IR SEDs, including AGN contribution.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://tinyurl.com/y9phjhxy",
    install_requires=['numpy>=1.19.4', 'matplotlib>=3.3.3', 'astropy>=4.2', 'scipy>=1.5.4', 'pandas>=1.1.5', 'emcee>=3.0.2', 'numba>=0.52.0', 'tqdm'],
    packages=['iragnsep'],
    package_data={'iragnsep': ['Filters/*.csv', 'iragnsep_templ.csv', 'ExtCurves/*.csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
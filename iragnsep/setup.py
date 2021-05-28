#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iragnsep",
    version="7.3.2",
    author="Emmanuel Bernhard",
    author_email="manu.p.bernhard@gmail.com",
    description="Fits of IR SEDs including AGN contribution.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://tinyurl.com/y9phjhxy",
    install_requires=['numpy>=1.20.3', 'matplotlib>=3.4.2', 'astropy>=4.2.1', 'scipy>=1.6.3', 'pandas>=1.2.4', 'emcee>=3.0.2', 'numba>=0.53.1', 'tqdm>=4.61.0'],
    packages=['iragnsep'],
    package_data={'iragnsep': ['Filters/*.csv', 'iragnsep_templ.csv', 'ExtCurves/*.csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
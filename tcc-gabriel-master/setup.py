#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:41:09 2019

@author: gabriel
"""

import setuptools

setuptools.setup(
    name="ps-search",
    version="0.0.1",
    author="Gabriel Antonio Carneiro",
    author_email="gabri14el@gmail.com",
    description="A tool for search similar histologial images using CNNs.",
#    long_description=long_description,
#    long_description_content_type="text/markdown",
    url="http://pathospotter.uefs.br/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
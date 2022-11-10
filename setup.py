# -*- coding: utf-8 -*-

import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()


    
setuptools.setup(
    name="GOShlearnpy",
    version="0.0.1",
    author="Hyoseob Noh",
    author_email="hyoddubi@naver.com",
    description="GOShLearnpy: Python library for Global Optmization and Shallow machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hyoddubi1/GOShLearnpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'numpy>=1.20.0',
        'scikit-learn',
        'matplotlib',  
        'pyDOE',
        'tqdm'
    ]
)
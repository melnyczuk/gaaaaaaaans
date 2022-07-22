#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="cloudput",
    version="0.0.1",
    description="a GAN for generating images of clouds",
    author="How Melnyczuk",
    author_email="h.melnyczuk@gmail.com",
    url="https://github.com/melnyczuk/cloudput",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    keywords=["gan", "pose", "posenet"],
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy==1.20.3",
        "torch==1.8.1",
    ],
    python_requires=">=3.10",
    zip_safe=False,
    include_package_data=True,
)

#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="llm-algo-search",
    version="0.0.1",
    description="Search for novel algorithm implementations using LLMs",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    url="https://github.com/awentzonline/llm-algo-search",
    install_requires=[
        "beautifulsoup4",
        "hydra-core",
        "torch",
        "numpy",
    ],
    packages=find_packages(),
)

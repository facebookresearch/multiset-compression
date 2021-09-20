"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import re

from setuptools import find_packages, setup

# from https://github.com/facebookresearch/ClassyVision/blob/master/setup.py
# get version string from module
with open(
    os.path.join(os.path.dirname(__file__), "multiset_codec/__init__.py"), "r"
) as f:
    readval = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if readval is None:
        raise RuntimeError("Version not found.")
    version = readval.group(1)
    print("-- Building version " + version)

with open("README.md", encoding="utf8") as f:
    readme = f.read()

# alphabetical order
install_requires = [
    "craystack @ https://github.com/j-towns/craystack/tarball/master#egg=20b278f92d6e947dcf71e75f8a44468c3019e076"
]

setup(
    name="multiset-codec",
    version=version,
    description="Compressing Multisets with Large Alphabets",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="Facebook AI Research",
    author_email="karenu@fb.com",
    license="MIT",
    project_urls={
        "Source": "https://github.com/facebookresearch/multiset-codec",
    },
    python_requires=">=3.6",
    setup_requires=["wheel"],
    install_requires=install_requires,
    packages=find_packages(
        exclude=[
            "tests",
            "data",
            "experiments",
            "figures"
        ]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Archiving :: Compression",
    ],
)

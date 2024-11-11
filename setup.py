"""Setup script for the HandoverOptimDRL package."""

from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="HandoverOptimDRL",
    version="0.1.0",
    description="Framework for learning handover algorithms using deep reinforcement learning.",
    author="Your Name",
    author_email="johannes.voigt@kit.edu",
    url="https://github.com/kit-cel/HandoverOptimDRL",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)

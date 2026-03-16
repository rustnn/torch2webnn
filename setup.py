"""Setup script for webnn_torch_export"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="webnn_torch_export",
    version="0.1.0",
    author="Maximilian Mueller",
    author_email="maximilianm@nvidia.com",
    description="Custom PyTorch exporter using Dynamo for WebNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/webnn_torch_export",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "webnn-export=webnn_torch_export.exporter:main",
        ],
    },
)

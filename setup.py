from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="skilltrain",
    version="1.0.0",
    author="SMF Works",
    author_email="michael@smfworks.com",
    description="Train Your Agent Skills Like Neural Networks — an OpenClaw-native skill optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smfworks/skilltrain",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "requests>=2.28",
    ],
    extras_require={
        "dev": ["pytest", "black"],
    },
    entry_points={
        "console_scripts": [
            "skilltrain=skilltrain.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

from setuptools import setup, find_packages

setup(
    name="binGPT",  # Package name
    version="0.1.0",  # Version number
    packages=find_packages(),  # Automatically find package(s) in the folder
    install_requires=[  # List dependencies here, e.g.,
        # 'numpy>=1.18.0',
    ],
    author="",
    author_email="",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",  # Minimum Python version
)

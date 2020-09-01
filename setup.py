import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slickml",
    version="0.0.3",
    author="Amirhessam Tahmassebi, Trace Smith",
    author_email="amir.benny@gmail.com, tsmith5151@gmail.com",
    description="slickml is a machine learning library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

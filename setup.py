from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="pytriplet",
    version="0.0.0.1",
    author="Congcong Wang",
    author_email="wangcongcongcc@gmail.com",
    description="Fine-tuning Transformers with TPUs or GPUs acceleration, written in Tensorflow2.0",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/wangcongcong123/ttt",
    download_url="https://github.com/wangcongcong123/ttt/releases/download/v0.0.0.1/pytriplet.tar.gz",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.3.0",
        "sklearn",
        "tqdm",
        "keras",
        "tensorboardX",
        "nlp",
        "sacrebleu",
        "transformers==3.1.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformers, Tensorflow, TPUs acceleration"
)
# commands for uploading to pypi
# python setup.py sdist
# pip install twine
# twine upload dist/*


from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="ttt",
    version="0.0.1",
    author="Congcong Wang",
    author_email="wangcongcongcc@gmail.com",
    description="Fine-tuning Transformers with TPUs or GPUs acceleration, written in Tensorflow",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="TBA",
    download_url="TBA",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.3.0",
        "sklearn",
        "tqdm",
        "keras",
        "tensorboardX",
        "nlp",
        "sacrebleu",
        "transformers"
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
# pip install -e .
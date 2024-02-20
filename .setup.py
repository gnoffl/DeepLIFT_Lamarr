from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

VERSION = '0.0.1'
DESCRIPTION = 'Deeplift algorithm to approximate shapley values in neural networks'

# Setting up
setup(
    name="DeepLIFT_Lamarr",
    version=VERSION,
    author="Gernot Schmitz",
    author_email="<gernot.schmitz@tu-dortmund.de>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'scikit-learn',
    ],
    keywords=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
from setuptools import setup, find_packages
import sys

with open("requirements.txt", 'r') as ifile:
    requirements = ifile.read().splitlines()

nb_requirements = [
    'nbconvert>=6.0.0',
    'nbformat>=5.1.0',
    'jupyter>=4.0.0',
    'ipython>=7.1.1'
]

setup(
    name="pmf",
    version="1.0.0",
    description="Parallel Probablistic Matrix Factorization",
    authors=["Anders Geil", "Sol Park", "Yinuo Jin"],
    packages=find_packages(),
    install_requires=requirements,
    package_data={'': ['bin/main.tsk']},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python ::3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require={'notebooks': nb_requirements},
)
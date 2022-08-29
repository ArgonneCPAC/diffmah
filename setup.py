import os
from setuptools import setup, find_packages


__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "diffmah", "_version.py"
)
with open(pth, "r") as fp:
    exec(fp.read())


setup(
    name="diffmah",
    version=__version__,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Differentiable model of dark matter halo assembly",
    install_requires=["numpy", "jax"],
    packages=find_packages(),
    url="https://github.com/ArgonneCPAC/diffmah",
    package_data={"diffmah": ("data/*.dat", "tests/testing_data/*.dat")},
)

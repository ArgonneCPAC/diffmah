from setuptools import setup, find_packages


PACKAGENAME = "diffmah"
VERSION = "0.1.0"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Differentiable model of dark matter halo assembly",
    long_description="Differentiable model of dark matter halo assembly",
    install_requires=["numpy", "jax"],
    packages=find_packages(),
    url="https://github.com/aphearin/diffmah",
    package_data={"diffmah": ("data/*.dat", "tests/testing_data/*.dat")},
)

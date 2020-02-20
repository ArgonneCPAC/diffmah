from setuptools import setup, find_packages


PACKAGENAME = "diffmah"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Some package",
    long_description="Just some package",
    install_requires=["numpy"],
    packages=find_packages(),
    url="https://github.com/aphearin/diffmah"
)

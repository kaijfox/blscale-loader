from setuptools import setup, find_packages

setup(
    name='blscale_loader',
    version='0.0.1',
    packages=find_packages(include=["blscale_loader", 'blscale_loader.*']),
)
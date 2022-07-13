from setuptools import setup, find_packages
from os import path

with open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().split('\n')

install_requires = [x.strip() for x in requirements]

setup(
    name='mlmodels',
    version='0.0.1',
    description='Implementation of several machine learning models from scratch using Python and NumPy.',
    url='https://github.com/rixsilverith/machine-learning-models',
    download_url='https://github.com/rixsilverith/machine-learning-models/tarball/main',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='Rix Silverith',
    author_email='rixsilverith@outlook.com',
    install_requires=install_requires,
    setup_requires=['numpy>=1.10', 'scipy>=0.17']
)

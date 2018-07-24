"""
A setuptools based setup module.

Adopted from:
https://github.com/pypa/sampleproject

"""

import codecs
from os import path

from setuptools import setup, find_packages


def get_long_description():
    """Load description from README file."""
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, 'README.rst'), encoding='utf-8') as file:
        return file.read()


setup(
    name='xentica',
    version='0.1.1',
    description='GPU-accelerated engine for multi-dimensional '
                'cellular automata',
    long_description=get_long_description(),
    url='https://github.com/a5kin/xentica',
    author='Andrey Zamaraev',
    author_email='a5kin@github.com',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Scientific/Engineering :: Artificial Life',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Code Generators',
        'Topic :: Software Development :: Libraries',
        'Topic :: Artistic Software',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    keywords='cellular automata ca simulation artificial life alife energy '
             'framework gpu multidimensional microprograms conservation',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'numpy',
        'cached-property',
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)

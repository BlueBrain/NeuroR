"""Morph-repair setup.py"""
import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")


VERSION = imp.load_source("morph_repair.version", "morph_repair/version.py").VERSION


setup(
    description='A morphology repair tool',
    author='BBP Neuroscientific Software Engineering',
    version=VERSION,
    install_requires=[
        'click>=0.7.0',
        'scipy>=1.1.0',
        'morphio>=2.1.1',
        'neurom @ git+https://git@github.com/wizmer/NeuroM.git@mut_morphio#egg=neurom-2.0.0',
        'plotly_helper>=0.0.1',
        'matplotlib>=2.2.3',
        'morph-tool>=0.1.14',
        'pathlib2>=2.3.3',
        'morph-tool>=0.1.14',
        'pandas>=0.24.2',
        'cut-plane>=0.0.6',
    ],
    packages=find_packages(),
    license="BBP-internal-confidential",
    name='morph-repair',
    entry_points={'console_scripts': ['morph-repair=morph_repair.cli:cli']},
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)

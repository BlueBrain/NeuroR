"""Morph-repair setup.py"""
import os
import imp
from setuptools import setup
from setuptools import find_packages


VERSION = imp.load_source("morph_repair.version", "morph_repair/version.py").VERSION

REQS = ['click>=0.7.0',
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
]


config = {
    'description': 'A morphology repair tool',
    'author': 'BBP Neuroscientific Software Engineering',
    'version': VERSION,
    'install_requires': REQS,
    'packages': find_packages(),
    'license': 'BSD',
    'name': 'morph-repair',
    'entry_points': {'console_scripts': ['morph-repair=morph_repair.cli:cli']},
}

setup(**config)

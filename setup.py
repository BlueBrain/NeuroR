"""Repair setup.py"""
import os
from setuptools import setup
from setuptools import find_packages


VERSION = "0.1.0"

REQS = ['click>=0.7.0',
        'scipy>=1.1.0',
        'morphio>=2.0.0',
        'neurom[plotly] @ git+https://git@github.com/wizmer/NeuroM.git@mut_morphio#egg=neurom-2.0.0',
        'matplotlib>=2.2.3',
        'cut-plane'
]


config = {
    'description': 'A morphology repair tool',
    'author': 'BBP Neuroscientific Software Engineering',
    'version': VERSION,
    'install_requires': REQS,
    'packages': find_packages(),
    'license': 'BSD',
    'name': 'repair',
    'entry_points': {'console_scripts': ['repair=repair.cli:cli']},
}

setup(**config)

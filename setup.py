"""neuror setup.py"""
import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")


VERSION = imp.load_source("neuror.version", "neuror/version.py").VERSION


setup(
    description='A morphology repair tool',
    author='bbp-ou-nse',
    version=VERSION,
    install_requires=[
        'click>=0.7.0',
        'matplotlib>=2.2.3',
        'morph-tool>=0.1.14',
        'morphio>=2.1.1',
        'neurom @ git+https://git@github.com/BlueBrain/NeuroM.git@mut_morphio#egg=neurom-2.0.0',
        'pandas>=0.24.2',
        'pyquaternion>=0.9.2',
        'scipy>=1.2.0',
    ],
    extras_require={
        'plotly': [
            'dash-core-components>=0.46.0',  # HTML components
            'dash-table>=3.6.0',  # Interactive DataTable component (new!)
            'dash>=0.41.0',  # The core dash backend
            'plotly-helper>=0.0.2',
        ],
        'docs': ['sphinx', 'sphinx-click', 'sphinx-bluebrain-theme'],
    },
    packages=find_packages(),
    license="BBP-internal-confidential",
    name='NeuroR',
    entry_points={'console_scripts': ['neuror=neuror.cli:cli']},
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

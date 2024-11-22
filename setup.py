from setuptools import setup, find_packages

setup(
    name='noda',
    version='2.1.0',
    description='Neutrino Oscillation Data Analysis: fitting framework for JUNO',
    author='Sindhujha Kumaran',
    author_email='s.kumaran@uci.edu',
    packages = find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'run_noda = noda.main:main',
        ],
    },

    install_requires=[
        'matplotlib',
        'scipy',
        'joblib',
        'numpy',
        'pandas',
        'emcee',
        'ptemcee',
        'xarray',
        'arviz',
        'corner',
        'iminuit',
        'uproot',
        'h5py',
    ],
    python_requires='>=3.10',
)

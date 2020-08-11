"""
Boolean Network Simulation Tool
"""
from setuptools import find_packages, setup

dependencies = [
    'click', 'numpy>=1.12.0', 'scipy>=0.14.0', 'pyyaml', 'matplotlib', 'seaborn', 'pillow', 'ZODB', 'BTrees',
    'persistent', 'transaction']
extras = {
    'mpi': ['mpi4py']}

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='boolsi',
    version='1.0.3',
    url='https://github.com/boolsi/boolsi',
    license='MIT',
    author='Vladyslav Oles, Anton Kukushkin',
    author_email='kukushkin.anton@gmail.com, vladyslav.oles@wsu.edu',
    description='BoolSi is a tool for distributed simulations and analysis of Boolean networks',
    long_description='BoolSi is a command line tool for distributed simulations and analysis of '
                     'synchronous Boolean networks.\n\n'
                     'Read more at: https://github.com/boolsi/boolsi/blob/master/README.md',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'boolsi = boolsi.cli:cli',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ]
)

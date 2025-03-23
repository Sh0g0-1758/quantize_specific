from setuptools import setup

setup(
    name='quantize_specific',
    version='0.0.1',
    description='a python library that lets you quantize any layer in the model',
    url='https://github.com/Sh0g0-1758/quantize_specific',
    author='Shourya Goel',
    author_email='shouryagoel10000@gmail.com',
    license='Apache-2.0',
    packages=['quantize_specific'],
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.36.0',
        'datasets>=2.15.0',
        'numpy>=1.24.0',
        'tqdm>=4.65.0',
        'accelerate>=0.26.0',
        'zstandard>=0.21.0',
        'rouge_score'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
    ],
)

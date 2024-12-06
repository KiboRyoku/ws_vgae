from setuptools import setup
from setuptools import find_packages

setup(name='ws_vgae',
      description='Weight Sharing in Variational Graph Autoencoder',
      author='G. Salha-Galvan and J. Xu (2025)',
      install_requires=['networkx==2.6',
                        'numpy==2.0.2',
                        'scikit-learn==1.5.2',
                        'scipy==1.14.1',
                        'tensorflow==2.18.0'],
      package_data={'ws_vgae': ['README.md']},
      packages=find_packages())

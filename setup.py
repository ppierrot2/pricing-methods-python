from setuptools import setup, find_packages

setup(name='pricing_python',
      version='0.1',
      description='some pricing methods in Python',
      package=find_packages(),
      install_requires=['setuptools', 'numpy', 'sklearn']
      )
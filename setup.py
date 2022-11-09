from setuptools import setup, find_packages

setup(
    name='begin',
    version='0.1.0',
    packages=find_packages(include=['begin', 'begin.algorithms.*', 'begin.evaluators.*', 'begin.scenarios.*', 'begin.trainers.*'], exclude=['_build', 'docs', 'dist', 'build', 'begin.egg-info'])
)
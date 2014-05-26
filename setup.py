#!/usr/bin/env python

from setuptools import setup

setup(name='turbustat',
      version='0.0',
      description='Implementation of metrics for comparing spectral line data cubes.',
      author='Team AstroStat',
      author_email='koch.eric.w@gmail.com',
      url='http://github.com/Astroua/TurbuStat',
      packages=['turbustat'],
      requires=['numpy','astropy','scipy','skimage','pandas', 'matplotlib', 'statsmodels', 'sklearn', 'astrodendro']
     )
#!/usr/bin/env python

from distutils.core import setup

setup(name='q1physrl_env',
      version='1.0',
      entry_points={
          'console_scripts': [
          ]
      },
      description='Reinforcement learning environment for Quake 1 player physics',
      install_requires=['gym'],
      author='Matt Earl',
      packages=['q1physrl_env'])

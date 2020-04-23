#!/usr/bin/env python

from distutils.core import setup


setup(name='q1physrl',
      version='1.0',
      entry_points={
          'console_scripts': [
                'q1physrl_train = q1physrl.train:train',
                'q1physrl_make_demo = q1physrl.mkdemo:make_demo_entrypoint',
                'q1physrl_plot_all_checkpoints = q1physrl.analyse:plot_all_checkpoints',
                'q1physrl_make_speed_anim = q1physrl.vidtools:make_speed_anim',
          ]
      },
      description='A script for training the q1physrl_env environment',
      install_requires=['ray[rllib]', 'numpy', 'pandas', 'requests'],
      author='Matt Earl',
      packages=['q1physrl'])


#!/usr/bin/env python
# Copyright (c) 2020 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


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


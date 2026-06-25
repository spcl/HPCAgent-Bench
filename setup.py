#!/usr/bin/env python

from distutils.core import setup


setup(name='optarena',
      version='0.1',
      url='https://github.com/spcl/OptArena',
      author='SPCL @ ETH Zurich',
      author_email='yakupkoray.budanaz@inf.ethz.ch',
      description='OptArena',
      license='GPL-3.0-or-later',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
          "Operating System :: OS Independent",
      ],
      packages=['optarena', 'optarena.infrastructure'],
      python_requires='>=3.6')

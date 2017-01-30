from setuptools import setup, Extension
import numpy

setup (name = 'xfel', \
       version = '1.0', \
       packages  = ['xfel',
                    'xfel.core',
                    'xfel.orientaton',
                    'xfel.sampling',
                    'xfel.test',
                ],
       package_dir = {'xfel': './xfel'},
       requires = ['numpy', 'scipy']
)


from setuptools import setup, Extension
import numpy

setup (name = 'bxfel', \
       version = '1.0', \
       packages  = ['bxfel',
                    'bxfel.core',
                    'bxfel.orientation',
                    'bxfel.sampling',
                    'bxfel.test',
                ],
       package_dir = {'bxfel': './bxfel'},
       package_data={'bxfel':['orientation/resources/gauss/*.dat',
                              'orientation/resources/chebyshev/*.dat']},
       include_package_data=True,
       requires = ['numpy', 'scipy'],
       zip_safe=False
       
)


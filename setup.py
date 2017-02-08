from setuptools import setup, Extension
import numpy

xfelmodule = Extension('bxfel/_bxfel',
                          include_dirs = [numpy.get_include(),
                                       './bxfel/c/include'],
                           sources = ['./bxfel/c/grid.c',
                                      './bxfel/c/xfel.c'],
                           extra_compile_args = ['-Wno-return-type',
                                                 '-Wno-unused-variable',
                                                 '-Wno-unused-function',
                                                 '-Wno-unused-but-set-variable']
)

setup (name = 'bxfel', \
       version = '1.0', \
       packages  = ['bxfel',
                    'bxfel.core',
                    'bxfel.io',
                    'bxfel.model',
                    'bxfel.orientation',
                    'bxfel.inference'
                ],
       package_dir = {'bxfel': './bxfel'},
       package_data={'bxfel':['orientation/resources/gauss/*.dat',
                              'orientation/resources/chebyshev/*.dat',
                              'data/volumes/*.mrc']},
       include_package_data=True,
       requires = ['numpy', 'scipy'],
       zip_safe=False,
       ext_modules = [xfelmodule,]
       
)


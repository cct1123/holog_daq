import os
from distutils.core import setup
import pkg_resources
import setuptools
import pathlib

VERSION = '0.1'
## TEST CHANGE
def build_packages(base_dir, name_base):
    '''
    recusively find all the folders and treat them as packages
    '''
    arr = [name_base]
    for fname in os.listdir(base_dir):
        if os.path.isdir(base_dir+fname):
            '''
            ignore the hidden files
            '''
            if fname[0]=='.':
                continue
            name = '{}.{}'.format(name_base, fname)
            recursion = build_packages(base_dir+fname+'/', name)
            if len(recursion) != 0:
                [arr.append(rec) for rec in recursion]
    return arr

packages = build_packages('holog_daq/', 'holog_daq')

setup(name='holog_daq',
      version=VERSION,
      description='Software for data acquisition of the holography setup.',
      author='Grace E. Chesmore, UChicago Lab',
      author_email='chesmore@uchicago.edu',
      package_dir={'holog_daq':'holog_daq'},
      packages=packages,
      scripts=['scripts/poco_init.py3', 'scripts/synth_init.py3', 
        'scripts/plot_cross.py3', 'scripts/plot_cross_phase_no_quant3.py'],
     )


# # Install requirements
# with pathlib.Path('requirements.txt').open() as requirements_txt:
#     install_requires = [
#         str(requirement)
#         for requirement
#         in pkg_resources.parse_requirements(requirements_txt)
#     ]

# setuptools.setup(
#     install_requires=install_requires,
# )
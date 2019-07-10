from distutils.core import setup

    packages=find_packages(exclude=['ez_setup', 'tests', 'tests.*']),


setup(
    name='XtractMaps',
    version='0.1dev',
    packages=find_packages(exclude=['ez_setup', 'tests', 'tests.*']),
    data_files = [('', ['xtract_maps/city_index.json'])],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
)

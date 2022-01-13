from setuptools import find_packages,setup
setup(
    name='ml_but_explained',
    packages=find_packages(),
    version='0.1.0',
    description='A machine learning library where the focus is on you learning not the machine!',
    author='Daniel Droder',
    license='MIT',
    install_requires=['numpy','pandas']
)
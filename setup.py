from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('MotorModel/__init__.py').read(),
)[0]

setup(name='MotorModel',
      version=__version__,
      author='',
      author_email='',
      url='https://github.com/tuckerbabcock/MotorModel',
      license='',
      packages=[
          'MotorModel',
      ],
      install_requires=[
          'numpy>=1.21.4',
          'openmdao>=3.18.0',
          'mphys>=0.4.0',
          'omESP'
      ],
      classifiers=[
        "Programming Language :: Python"]
)

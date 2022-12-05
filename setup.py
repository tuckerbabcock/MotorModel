from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('motormodel/__init__.py').read(),
)[0]

setup(name='motormodel',
      version=__version__,
      author='Tucker Babcock',
      author_email='tuckerbabcock1@gmail.com',
      url='https://github.com/tuckerbabcock/MotorModel',
      license='BSD 3-Clause License',
      packages=[
          'motormodel',
      ],
      python_requires=">=3.8",
      install_requires=[
          'numpy>=1.21.4',
          'openmdao>=3.18.0',
          'mphys>=0.4.0',
          'omESP'
      ],
      classifiers=[
        "Programming Language :: Python"
      ]
)

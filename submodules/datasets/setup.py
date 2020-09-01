from setuptools import setup, find_packages
import os


def get_version():
    path = os.path.dirname(os.path.realpath(__file__))
    f = os.path.join(path, 'datasets', '__init__.py')
    with open(f, 'r') as f:
        for line in f:
            if not line.startswith('__version__'):
                continue
            v = line.strip().split('__version__ = ')[1].strip("'")
            return v
    raise ValueError('version not found')


setup(name='brain-datasets',
      version=get_version(),
      description='collection of datasets',
      # The project's main homepage.
      url='https://github.kakaobrain.com/kakaobrain/datasets',
      author='dade',
      author_email='dade.ai@kakaobrain.com',
      packages=find_packages(),
      install_requires=['torch', 'torchvision', 'scipy', 'xmltodict'],
      setup_requires=['setuptools'],
      include_package_data=True,
      )

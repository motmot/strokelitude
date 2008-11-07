from setuptools import setup, find_packages

setup(name='strokelitude',
      version='0.1',
      entry_points = {
    'motmot.fview.plugins':
    'strokelitude = strokelitude.strokelitude:StrokelitudeClass',
    },

      packages = find_packages(),
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      zip_safe=True,
      package_data = {'strokelitude':['strokelitude.xrc']},
      )

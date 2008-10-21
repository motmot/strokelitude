from setuptools import setup, find_packages

setup(name='strokelitude_stripe',
      version='0.1',
      entry_points = {
    'motmot.fview_strokelitude.plugins':'motmot.fview_strokelitude.plugins = stripe.stripe:StripeClass',
    },
      packages = find_packages(),
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      )

from setuptools import setup, find_packages

setup(name='strokelitude_stripe',
      version='0.1',
      entry_points = {
    'strokelitude.plugins':[
    'StripePluginInfo = stripe.experiment_runner:StripePluginInfo',
    ## 'StripePluginInfo = stripe.stripe:StripePluginInfo',
    ## 'SequencePluginInfo = stripe.sequence:SequencePluginInfo',
    ],
    },
      packages = find_packages(),
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      )

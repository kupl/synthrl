from setuptools import setup, find_packages

setup_params = dict(
  name='SynthRL',
  version='0.0',
  description='Program Synthesizer using Reinforcement Learning',
  url='https://github.com/kupl/SynthRL-dev',
  author='Software Analysis Labrotory at Korea University',
  author_email='noemail@no.email',
  packages=find_packages(exclude=['example']),
  setup_requires=[    # required library for setup.py itself
  ], 
  install_requires=[  # required library for SynthRL
    'lark-parser',
    'numpy'
  ], 
  dependency_links=[  # libraries cannot be found on pip
  ],
)

if __name__ == '__main__':
  setup(**setup_params)

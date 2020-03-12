import logging

from synthrl.env import SynthesizerEnvironment
from synthrl.env import VerifierEnvironment
from synthrl.utils import Timer

logger = logging.getLogger(__name__)

def synthesize_from_oracle(dsl=None, synthisizer=None, verifier=None, oracle=None, ioset=[], budget=None):
  # gets callable dsl
  #      synthsizer and verifier agent
  #      callable oracle and initial ioset
  #      time budget
  # and returns synthesized program
  trail = 0
  for t in Timer(budget):
    trail += 1
    logger.info('[{:.2f}s] {} trails'.format(t.total_seconds(), trail))

def synthesize_interactively(dsl=None, synthesizer=None, verifier=None):
  raise NotImplementedError

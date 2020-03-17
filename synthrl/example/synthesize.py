import logging

from synthrl.env import MAEnvironment
from synthrl.utils import Timer

logger = logging.getLogger(__name__)

def synthesize_from_oracle(dsl=None, synthesizer=None, verifier=None, oracle=None, ioset=[], budget=None, testing=None, testing_opt={}):
  # gets callable dsl
  #      synthsizer and verifier agent
  #      callable oracle and initial ioset
  #      time budget
  # and returns synthesized program
  trail = 0
  for t in Timer(budget):
    trail += 1
    logger.info('[{:.2f}s] {} trails'.format(t.total_seconds(), trail))

    env = MAEnvironment(ioset=ioset, dsl=dsl, testing=lambda pgm1, pgm2: testing(pgm1, pgm2, **testing_opt))
    state, _, (t_syn, t_ver) = env.reset()

    while not t_syn:
      action = synthesizer.take(state, env.action_space)
      state, _, (t_syn, t_ver) = env.step(action)

    while not t_ver:
      action = verifier.take(state, env.action_space)
      state, _, (_, t_ver) = env.step(action)

    # update ioset

def synthesize_interactively(dsl=None, synthesizer=None, verifier=None):
  raise NotImplementedError

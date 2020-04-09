import logging

from synthrl.env import MAEnvironment
from synthrl.utils import TimeoutException
from synthrl.utils import Timer

logger = logging.getLogger(__name__)

def synthesize_from_oracle(dsl=None, synthesizer=None, verifier=None, oracle=None, ioset=[], budget=None, testing=None, testing_opt={}):
  # gets callable dsl
  #      synthsizer and verifier agent
  #      callable oracle and initial ioset
  #      time budget
  # and returns synthesized program
  trail = 0
  program = dsl()
  timer = Timer(budget)
  try:
    while True:
      logger.info('[{:.2f}s] {} trails'.format(timer.elapsed.total_seconds(), trail))
      trail += 1
      synthesizer.reset()
      verifier.reset()

      env = MAEnvironment(ioset=ioset, dsl=dsl, testing=lambda pgm1, pgm2: testing(pgm1, pgm2, **testing_opt))
      env.set_timer(timer)
      state, _, (t_syn, t_ver) = env.reset()

      while not t_syn:
        action = synthesizer.take(state, env.action_space)
        state, _, (t_syn, t_ver) = env.step(action)
      
      logger.debug('{} trails: program synthesized.'.format(trail))
      program = env.program

      while not t_ver:
        action = verifier.take(state, env.action_space)
        state, _, (_, t_ver) = env.step(action)

      distinguishing_input = env.distinguishing_input
      ## logging ##
      print('--distingushing--')
      print(distinguishing_input)
      ## logging ##
      try:
        ioset.append((distinguishing_input, oracle(*distinguishing_input)))
      except:
        pass
  except TimeoutException:
    logger.info('Timeout: {} trails'.format(trail))
  
  return program
  
def synthesize_interactively(dsl=None, synthesizer=None, verifier=None):
  raise NotImplementedError

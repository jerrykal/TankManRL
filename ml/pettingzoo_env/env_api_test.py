import numpy as np
import supersuit as ss
from pettingzoo.test import api_test

import tankman

env = tankman.env(1, 1, 1000)
# env = ss.normalize_obs_v0(env)

for _ in range(100):
    api_test(env, num_cycles=1000, verbose_progress=True)

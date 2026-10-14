import numpy as np
from pup.policy.mlp_numpy import NumpyPolicy
p = NumpyPolicy.load("results/pup_policy.npz")
print(p.obs_size, p.action_size, p(np.zeros(45)).shape)   # 45 12 (12,)
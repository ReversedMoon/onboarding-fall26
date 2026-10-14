
from brax.io import model
from pup.train.export import export_policy
params = model.load_params("runs/smoke/policy.pkl")
export_policy(params[1], params[0], "runs/smoke/policy.npz")

import numpy as np
from pup.policy.mlp_numpy import NumpyPolicy
p = NumpyPolicy.load("runs/smoke/policy.npz")
print(p(np.zeros(45)).shape, p.joint_targets(np.zeros(45)).shape)   # (12,) (12,)
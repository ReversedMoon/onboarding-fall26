from brax.io import model
from pup.train.export import export_policy

params = model.load_params("results/policy.pkl")
out = export_policy(params[1], params[0], "results  /pup_policy.npz")
print("wrote", out)

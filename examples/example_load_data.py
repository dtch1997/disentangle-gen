# A short script showing how to load data from disentanglement_lib. 

# First, make sure to do: source CS236/bin/activate

import numpy as np
from disentanglement_lib.data.ground_truth import cars3d

data = cars3d.Cars3D()
randstate = np.random.RandomState()
# Sampling factors
factors = data.sample_factors(4, randstate)
observations = data.sample_observations_from_factors(factors, randstate)

print("Factors: {}".format(factors))
print("observations: {}".format(observations))


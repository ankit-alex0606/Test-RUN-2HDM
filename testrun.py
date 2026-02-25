%matplotlib inline
import numpy as np
import thdm_model


#this is a test run to see the values of Vtot at high temperature, and it seems that Vtot([0,0],T=500) is not the global minima in my case,
  # please check if you are getting similar results as me in your run

beta = 1.2490457723982544

model = thdm_model.model(493835.80266761663,    #one set of parameters taken from pt_scan.py
    47926.200296401825,
    167216.1008892055,
    0.5730380944516162,
    0.2620831167441152,
    11.628891483838677,
    -5.702838869654002,
    -5.702838869654002,
    np.sin(beta),
    246,
    300
)

import numpy as np


T = 500


h1_vals = np.linspace(-500, 500, 5)
h2_vals = np.linspace(-500, 500, 5)

for h1 in h1_vals:
    for h2 in h2_vals:
        V = model.Vtot([h1, h2], 500)
        print(f"V({h1:7.1f}, {h2:7.1f}) -> {V: .4e}")
    print("-" * 60)

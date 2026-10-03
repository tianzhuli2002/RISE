# RISE R0 empirical classifier objective

The reweighted adapter selects source EVENT_ID uniformly, theta uniformly over
the nine grid points, and membership independently with probability one half.
For membership 1 it supplies target importance weight w_i(theta); for membership
0 it supplies reference weight w_i(theta0). The BCE multiplier is that weight
divided by the complete training-population component mass divided by N. The
reference mass/population is 1. The target masses/population are:

- theta grid index 0: target mass/population = 1.00209728905
- theta grid index 1: target mass/population = 1.0090540604
- theta grid index 2: target mass/population = 1.00453972977
- theta grid index 3: target mass/population = 0.99895759625
- theta grid index 4: target mass/population = 1
- theta grid index 5: target mass/population = 0.999500898171
- theta grid index 6: target mass/population = 0.993472218597
- theta grid index 7: target mass/population = 0.987991445894
- theta grid index 8: target mass/population = 0.975010930181

Consequently the unchanged expected minibatch risk is

    L(f) = (1/(2G)) sum_theta [ sum_i w_i(theta)/W_theta BCE(f(x_i,theta),1)
                                + sum_i w_i(theta0)/W_0 BCE(f(x_i,theta),0) ].

The factor one half comes from equal membership sampling; 1/G comes from the
uniform theta sampler. The normalized component measures have equal prior mass.
For a continuous source measure q(x), its unrestricted Bayes logit is therefore

    log [ q(x) w_theta(x)/W_theta ] - log [ q(x) w_0(x)/W_0 ]
      = log r(x;theta,theta0) + log(W_0/W_theta).

Thus the finite-sidecar optimum includes c(theta)=log(W_0/W_theta). For the
analytic R0 density-ratio weights, W_theta/N estimates one and W_0/N=1, so this
offset vanishes in the infinite population. The reported exact-basis gradient
and coefficient uncertainty test whether finite-sidecar fluctuations explain
the observed empirical offset. There is no extra class-prior term because
membership priors are equal, and no theta-prior term survives because both
membership classes receive the same independently sampled theta. Signed
weights are nonnegative in this unsigned R0 sidecar; the code rejects negative
weights. In the full deterministic diagnostic, uniform theta and membership
sampling are integrated by enumerating every grid point and both classes once.
